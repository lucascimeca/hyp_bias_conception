import torch
import matplotlib.pyplot as plt
import networkx as nx
from utils.simple_io import *
from sklearn.decomposition import PCA
from utils.results_utils import *


TENSORBOARD_FOLDER = "./../runs/"
RESULTS_FOLDER = "./../results/generated/"

# ['#e52592', '#425066', '#12b5cb', '#f9ab00', '#9334e6', '#7cb342', '#e8710a']
feature_to_color_dict = {
    "diagonal": '#e52592',
    "scale": '#425066',
    "color": '#12b5cb',
    "shape": '#9334e6',
    "orientation": '#e8710a',
    "object_number": '#7cb342',
    "x_position": '#62733c',
    "age": '#ad63a8',
    "gender": '#ff0004',
    "ethnicity": '#7a7a7a'
}


def pca_2d_weight_distribution_figure(data_dict, overfit_container, pca_2d=None, colors=None):
    # ---- cumulative explained variance ------

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_title("2d weight distributions (p1, p2)")
    markers = ['o', 'x', '^']

    for i, experiment_name in enumerate(data_dict.keys()):
        data = data_dict[experiment_name].cpu().numpy()
        for j in range(data.shape[1]):
            X = data[:, j, :]
            data_reduced = pca_2d.transform(X)
            label = "'{}':{:.2f}%, sample {}".format(overfit_container[experiment_name][0][0],
                                                     overfit_container[experiment_name][0][1],
                                                     j)
            plt.scatter(data_reduced[:, 0], data_reduced[:, 1], marker=markers[j], color=colors[i],
                        label=label)

    ax.legend()
    ax.set_xlabel('PCA - $p_1$')
    ax.set_ylabel('PCA - $p_2$')
    plt.show()


def cumulate_explained_variance_figure(data_dict, overfit_container, colors=None):
    # ---- cumulative explained variance ------

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_title("Rank diversity of solutions (Explained Variance of solution weights)")

    for i, experiment_name in enumerate(data_dict.keys()):
        X = data_dict[experiment_name][:, 0, :].cpu().numpy()
        pca = PCA()  # all dimensions
        pca.fit(X)
        cumulative_explained_variance = [sum(pca.explained_variance_ratio_[:i]) * 100 for i in
                                         range(len(pca.explained_variance_ratio_))]
        label = "overfit to '{}' at {:.2f}%".format(overfit_container[experiment_name][0][0],
                                                    overfit_container[experiment_name][0][1])
        plt.plot(cumulative_explained_variance[:20], color=colors[i], label=label)

    ax.legend()
    ax.set_xlabel("Dimensions (Principal Components)")
    ax.set_ylabel("Explained Variance (%)")
    plt.show()


def bar_plot_figure(overfit_container):
    # ---- cumulative explained variance ------

    plt.figure(figsize=(4, 7))
    fig, axes = plt.subplots(1, 3)
    plt.title("Solutions overfit stability")

    for i, experiment_name in enumerate(overfit_container.keys()):
        ys = [x[1] for x in overfit_container[experiment_name]]
        xs = list(range(len(ys)))
        x_labels = [x[0] for x in overfit_container[experiment_name]]
        axes[i].bar(xs, ys, .5)
        axes[i].set_ylabel('damped oscillation')
        axes[i].set_xlabel(experiment_name)
        axes[i].set_ylabel('overfit ratio (%)')
        axes[i].set_xticks(xs)
        axes[i].set_xticklabels(x_labels, rotation=45)

    plt.show()


def compute_distance(arr1, arr2):
    diff = arr1-arr2
    return torch.sqrt(torch.dot(diff, diff)).cpu()


def graph_distance_figure(all_data, labels, colors=None):
    indeces = list(range(all_data.shape[0]))

    filename = "weight_distances"
    extension = ".npz"

    if not file_exists(RESULTS_FOLDER + filename + extension):
        print("Could not find distance_matrix for weight distances, generating...")
        distance_matrix = np.zeros((len(indeces), len(indeces)), dtype=np.float64)
        for i in range(len(indeces)):
            for j in range(len(indeces)):
                if i == j:
                    continue
                distance_matrix[i, j] = compute_distance(all_data[i, :], all_data[j, :])
                distance_matrix[j, i] = distance_matrix[i, j]
                p_id = i*len(indeces) + j
                if p_id % 100 == 0:
                    print("Weight distances generation, progress {:.2f}%".format((i*len(indeces) + j)*100/len(indeces)**2),
                          flush=True,
                          end='\r')
        np.savez_compressed(RESULTS_FOLDER + filename,
                            distance_matrix=distance_matrix)
    else:
        print("Found distance_matrix for weight distances, loading...")
        distance_matrix = np.load(RESULTS_FOLDER + filename + extension)
        distance_matrix = distance_matrix['distance_matrix']


    # --- Distance matrix plot -----

    plt.figure(0, (10, 10))
    fig, ax = plt.subplots(1, 1)
    plt.title('Distance Matrix', fontsize=14)
    plt.imshow(distance_matrix, cmap='hot')
    plt.colorbar()
    x_ticks = list(range(0, distance_matrix.shape[0], 50))
    label_list = [labels[i] for i in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(label_list)
    plt.xticks(rotation=45)
    ax.set_yticks(x_ticks)
    ax.set_yticklabels(label_list)
    plt.show()


    # G = nx.from_numpy_matrix(distance_matrix)
    # G_tmp = nx.drawing.nx_agraph.to_agraph(G)
    # G_tmp.draw(RESULTS_FOLDER + 'out.png', format='png', prog='neato')

    # plt.figure(0, (20, 20))
    # pos = nx.circular_layout(G)
    # nx.draw(G,pos, node_size=2, alpha=0.1, node_color='black')

    # plt.tight_layout()
    # plt.axis('equal')
    # plt.show()

    return True


if __name__ == "__main__":

    if not folder_exists(RESULTS_FOLDER):
        folder_create(RESULTS_FOLDER)

    print("retrieving solutions weights...")
    weight_data, label_overfit_dictionary = load_weight_data(TENSORBOARD_FOLDER)
    print("Done!")

    overfit_container = {}

    all_data = None
    all_labels = None
    exp_labels = None
    sample_labels = None

    for exp_idx, experiment_name in enumerate(weight_data.keys()):
        # attach solution weights validation curves
        data_tmp = weight_data[experiment_name].view(-1, weight_data[experiment_name].shape[-1])
        exp_tmp = torch.ones((weight_data[experiment_name].shape[:-1])).view(-1, 1) * exp_idx
        sample_tmp = torch.tensor(list(range(weight_data[experiment_name].shape[0]))).view(-1, 1).repeat(1, 3).view(-1,
                                                                                                                    1)
        overfit_tmp = np.array(label_overfit_dictionary[experiment_name]).reshape(-1, 1)
        if all_data is None:
            all_data = data_tmp
            exp_labels = exp_tmp
            sample_labels = sample_tmp
            all_labels = overfit_tmp
        else:
            all_data = torch.cat([all_data, data_tmp], dim=0)
            exp_labels = torch.cat([exp_labels, exp_tmp], dim=0)
            sample_labels = torch.cat([sample_labels, sample_tmp], dim=0)
            all_labels = np.concatenate([all_labels, overfit_tmp], axis=0)

        overfit_container[experiment_name] = []
        for label in np.unique(label_overfit_dictionary[experiment_name]):
            overfit_container[experiment_name] += [
                (label, np.sum((overfit_tmp == label).astype(np.int)) * 100 / np.prod(overfit_tmp.shape))]
        overfit_container[experiment_name] = sorted(overfit_container[experiment_name], key=lambda x: x[1], reverse=True)

    all_data = all_data.view(-1, all_data.shape[-1])
    exp_labels = exp_labels.view(-1, 1)
    sample_labels = sample_labels.view(-1, 1)
    all_labels = all_labels.reshape(-1, 1)

    pca_2d = PCA(n_components=2)
    pca_2d.fit(all_data)

    # ------------------- PLOT FIGURES ------------------

    colors = ['#e52592', '#425066', '#12b5cb']

    # FIG 0
    bar_plot_figure(overfit_container)

    # FIG 1
    cumulate_explained_variance_figure(weight_data,
                                       overfit_container=overfit_container,
                                       colors=colors)

    # FIG 2
    pca_2d_weight_distribution_figure(weight_data,
                                      overfit_container=overfit_container,
                                      pca_2d=pca_2d,
                                      colors=colors)

    # FIG 3
    graph_distance_figure(all_data=all_data,
                          labels=all_labels,
                          colors=colors
                          )
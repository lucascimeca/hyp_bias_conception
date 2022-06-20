# distinctive_expert_embedding

* Code for different models in "./Models/"
* ./runs/folder created at run time to keep the logs of the runs for each experiment
* Run "main.py" for main experiments, running the ranking of models for each cue










# to run batch of experiments on nsml

# --- feature combinations
- nsml run -d BWDSprites --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -d ColorDSpritesPruned --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -d MultiDSprites --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -d MultiColorDSprites --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -d UTKFace --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -d UTKFace2 --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000


# --- solution state preturbation save (multiple experiments with different sets of feature cues)
- nsml run -e main_perturbation.py -d ColorDSpritesPruned --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000

# --- solution state augmentaion save (multiple experiments with same set of feature cues but augmentations by different cues each time)
- nsml run -e main_augmentation.py -d ColorDSpritesPruned --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -e main_augmentation.py -d UTKFace --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000


# --- observe loss around local minima solutions
- nsml run -e main_surf_test.py -d ColorDSpritesPruned --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -e main_surf_test.py -d UTKFace --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000


- nsml run -e main_sphere_test.py -d ColorDSpritesPruned --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -e main_sphere_test.py -d UTKFace --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000


# --- mode connectivity code

- nsml run -e main_mode_connectivity.py -d ColorDSpritesPruned --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -e main_mode_connectivity.py -d UTKFace --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000

# --- feature radius rerun code

- nsml run -e main_radius_rerun.py -d ColorDSpritesPruned --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -e main_radius_rerun.py -d UTKFace --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000

# --- feature test depth code

- nsml run -e main_test_minimal_ffnet.py -d ColorDSpritesPruned --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -e main_test_minimal_ffnet.py -d UTKFace --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000

# to retrieve tflogs 							  
- nsml download KR95157/ColorDSpritesPruned/26 Downloads -s /app/runs.zip
- nsml download KR95157/UTKFace/26 Downloads -s /app/runs.zip


# to push dataset
nsml dataset push ColorDSpritesPruned data/



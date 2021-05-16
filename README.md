# distinctive_expert_embedding

NSML COMMANDS


# to run batch of experiments on nsml

# --- feature combinations
- nsml run -d BWDSprites --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -d ColorDSprites --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -d MultiDSprites --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -d MultiColorDSprites --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -d UTKFace --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -d UTKFace2 --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000


# --- solution state preturbation save (multiple experiments with different sets of feature cues)
- nsml run -e main_perturbation.py -d ColorDSprites --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000

# --- solution state augmentaion save (multiple experiments with same set of feature cues but augmentations by different cues each time)
- nsml run -e main_augmentation.py -d ColorDSprites --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000



# to retrieve tflogs 							  
- nsml download KR95157/dsprites/SESS_NO Downloads -s /app/runs.zip

# to push dataset
nsml dataset push ColorDSprites data/


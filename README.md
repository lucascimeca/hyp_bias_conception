# distinctive_expert_embedding

NSML COMMANDS


# to run batch of experiments on nsml
- nsml run -d BWDSprites --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -d ColorDSprites --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -d MultiDSprites --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000
- nsml run -d MultiColorDSprites --gpu-driver-version 418.67 --cpus 6 --memory 40000000000 --shm-size 500000000


# to retrieve tflogs 							  
- nsml download KR95157/dsprites/SESS_NO Downloads -s /app/runs.zip

# to push dataset
nsml dataset push ColorDSprites data/


#!/bin/bash
####
#a) Define slurm job parameters
####
#SBATCH --job-name=tune

#resources:
#SBATCH --cpus-per-task=4
# the job can use and see 4 CPUs (from max 24).
#SBATCH --partition=gpu-2080ti
# the slurm partition the job is queued to.
#SBATCH --mem=8G

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)
#SBATCH --time=02:00:00
# the maximum time the scripts needs to run
# "hours:minutes:seconds"
#SBATCH --array=1-2

#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out
#SBATCH --mail-user=ludwig.bald@student.uni-tuebingen.de
# your mail address
####
#b) copy all needed data to the jobs scratch folder
####


####
#c) Execute the code
####

source $HOME/.bashrc

conda activate citylearn
python3 run_experiment.py $(head -${SLURM_ARRAY_TASK_ID} experiments/tuning/tuning.txt | tail -1) --run_id ${SLURM_ARRAY_TASK_ID}
conda deactivate


echo DONE!

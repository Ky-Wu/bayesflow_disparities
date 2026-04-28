#!/bin/bash
#$ -cwd #uses current working directory
# error = Merged with joblog
#$ -o joblog.$JOB_ID.$TASK_ID #creates a file called joblog.jobidnumber.taskidnumber to write to.
#$ -j y
#$ -l h_rt=20:00:00,h_data=4G #requests 20 hours, 4GB of data (per core)
#$ -pe shared 4 #requests 4 cores
# Email address to notify
#$ -M $USER@mail #don't change this line, finds your email in the system
# Notify when
#$ -m ea #sends you an email (b) when the job begins (e) when job ends (a) when job is aborted (error)
#$ -t 1-1:1 # 1 to 1, with step size of 1

. /u/local/Modules/default/init/modules.sh

module load anaconda3
module load gcc/11.3.0
module load openblas
module load cmake

conda activate bayesflow

# for logging
export TERM=dumb

echo "Running from: $(pwd)"

# Compute RDETs

echo "----------------------------------------------------------------"
echo "Computing RDETs for Reported Disparities"
echo "----------------------------------------------------------------"

python -u -m src.RDA_compute_RDETs

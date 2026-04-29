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

# 1. Setup Directories
# Get the directory of the current script, then move up to project root

echo "Running from: $(pwd)"

# 2. Clean and aggregate data

echo "----------------------------------------------------------------"
echo "Cleaning and setting up Lung Cancer Dataset"
echo "----------------------------------------------------------------"

python -u -m src.RDA_data_setup

# 3. Fit neural posterior estimator

echo "----------------------------------------------------------------"
echo "Fitting NPE to Lung Cancer Data"
echo "----------------------------------------------------------------"

python -u -m src.RDA_fit_joint_network

# 4. Analyze Dataset

echo "----------------------------------------------------------------"
echo "Applying ABI Network to Boundary Detection for Lung Cancer Data"
echo "----------------------------------------------------------------"

python -u -m src.RDA_analyze_data

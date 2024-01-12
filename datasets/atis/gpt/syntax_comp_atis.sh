#!/bin/bash
#SBATCH --account=p122-23-t  # project code
#SBATCH -J "gpt_synt_4"  # name of job in SLURM
#SBATCH --partition=short  # selected partition, ncpu or ngpu
#SBATCH --nodes=1  # number of used nodes
#SBATCH --time=18:00:00  # time limit for a job
#SBATCH -o atis/gpt/logs_syntax/synt_stdout.%J.out  # standard output
#SBATCH -e atis/gpt/logs_syntax/synt_stderr.%J.out  # error output

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate falcon-7b

#python -m bitsandbytes

python atis/gpt/compute_syntax_atis.py

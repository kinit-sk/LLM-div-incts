#!/bin/bash
#SBATCH --account=p122-23-t  # project code
#SBATCH -J "gpt_synt"  # name of job in SLURM
#SBATCH --partition=short  # selected partition, ncpu or ngpu
#SBATCH --nodes=1  # number of used nodes
#SBATCH --time=20:00:00  # time limit for a job
#SBATCH -o ag_news/gpt/logs_syntax/synt_stdout.%J.out  # standard output
#SBATCH -e ag_news/gpt/logs_syntax/synt_stderr.%J.out  # error output

# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate falcon-7b

#python -m bitsandbytes

cd $HOME

for i in 1 2 3 4 5; do
	python ag_news/gpt/compute_syntax_gpt.py --Iteration=0
	python ag_news/gpt/compute_syntax_gpt.py --Iteration=1
	python ag_news/gpt/compute_syntax_gpt.py --Iteration=2
	python ag_news/gpt/compute_syntax_gpt.py --Iteration=3
	python ag_news/gpt/compute_syntax_gpt.py --Iteration=4

done

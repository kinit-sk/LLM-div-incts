#!/bin/bash
#SBATCH --account=p122-23-t  # project code
#SBATCH -J "train_bert_atis_chain_gpt"  # name of job in SLURM
#SBATCH --partition=gpu  # selected partition, ncpu or ngpu
#SBATCH --gres=gpu:1  # total gpus
#SBATCH --nodes=1  # number of used nodes
#SBATCH --time=23:00:00  # time limit for  job
#SBATCH -o atis/gpt/logs_chain/bert_stdout.%J.out  # standard output
#SBATCH -e atis/gpt/logs_chain/bert_stderr.%J.out  # error output


# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate falcon-7b
module load cuda/12.0.1

cd $HOME

for i in 1 2 3 4 5; do
	python atis/gpt/train_bert_atis.py --Iteration=0
	python atis/gpt/train_bert_atis.py --Iteration=1
	python atis/gpt/train_bert_atis.py --Iteration=2
	python atis/gpt/train_bert_atis.py --Iteration=3
	python atis/gpt/train_bert_atis.py --Iteration=4
done


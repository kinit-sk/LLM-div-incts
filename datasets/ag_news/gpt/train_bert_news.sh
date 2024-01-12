#!/bin/bash
#SBATCH --account=p122-23-t  # project code
#SBATCH -J "train_bert_news_under_hints"  # name of job in SLURM
#SBATCH --partition=gpu  # selected partition, ncpu or ngpu
#SBATCH --gres=gpu:1  # total gpus
#SBATCH --nodes=1  # number of used nodes
#SBATCH --time=20:00:00  # time limit for  job
#SBATCH -o ag_news/gpt/logs_hints/bert_stdout.%J.out  # standard output
#SBATCH -e ag_news/gpt/logs_hints/bert_stderr.%J.out  # error output


# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate falcon-7b
module load cuda/12.0.1

cd $HOME

for i in 1 2 3 4 5; do
	python ag_news/gpt/train_bert_news.py --Iteration=0
	python ag_news/gpt/train_bert_news.py --Iteration=1
	python ag_news/gpt/train_bert_news.py --Iteration=2
	python ag_news/gpt/train_bert_news.py --Iteration=3
	python ag_news/gpt/train_bert_news.py --Iteration=4
done


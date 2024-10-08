# Effects of diversity incentives on sample diversity and downstream model performance in LLM-based text augmentation

This is data for the paper: ["Effects of diversity incentives on sample diversity and downstream model performance in LLM-based text augmentation"](https://arxiv.org/abs/2401.06643) published at [ACL'24](https://2024.aclweb.org/) main.

## Abstract

The latest generative large language models (LLMs) have found their application in data augmentation tasks, where small numbers of text samples are LLM-paraphrased and then used to fine-tune the model. However, more research is needed to assess how different prompts, seed data selection strategies, filtering methods, or model settings affect the quality of paraphrased data (and downstream models). In this study, we investigate three text diversity incentive methods well established in crowdsourcing: taboo words, hints by previous outlier solutions, and chaining on previous outlier solutions. Using these incentive methods as part of instructions to LLMs augmenting text datasets, we measure their effects on generated texts' lexical diversity and downstream model performance. We compare the effects over 5 different LLMs and 6 datasets. We show that diversity is most increased by taboo words, while downstream model performance is highest when previously created paraphrases are used as hints.


# Files

**Important note:** We provide a ``requirements.txt`` file for our setup and pytorch on CPU(!): please note that if you wish to run these experiments on your GPU to have the pytorch version conforming to your version of CUDA installed on your machine - details can be found [here](https://pytorch.org/get-started/locally/).

### Files

The directories in ``datasets`` contain collected data for each of the datasets used in the studies, together with example ``.py`` scripts that were used to collect data or train BERT classifiers on the data. For the OpenAI models (ChatGPT and GPT-4) data collection jupyter notebooks are provided.

The directory ``results`` contains ``.xlsx`` files with performance results of BERT classfiers trained on the collected data. 


### Paper citing

```
@inproceedings{cegin-etal-2024-effects,
    title = "Effects of diversity incentives on sample diversity and downstream model performance in {LLM}-based text augmentation",
    author = "Cegin, Jan  and
      Pecher, Branislav  and
      Simko, Jakub  and
      Srba, Ivan  and
      Bielikova, Maria  and
      Brusilovsky, Peter",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.710",
    pages = "13148--13171"
}

```

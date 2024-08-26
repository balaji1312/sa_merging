# Selective Attention Merging for low resource tasks: A case study of Child ASR

## Overview

This github repository contains models, scripts and data splits from our paper submitted to ICASSP 2025

## Folder Structure

Source code for our training and evaluating models  can be found under /src

Code for our proposed SA Merge can be found under /merge

/egs contains utterance ids for train/test splits for the MyST and CMU Kids databases

## Getting Started

1. **Install Dependencies**: transformers==4.32.1 torch evaluate datasets
2. On older versions of transformers it might be necessary to make minor edits to trainer.py to allow hotloading of Iterable datasets (if streaming is set to True). Follow the instructions in /egs/MyST/README.txt to make the necessary edits.
3. For training SSL based models, it is necessary to clone [the fairseq github repo](https://github.com/facebookresearch/fairseq)
4. To train/evaluate a model on a particular dataset, edit the corresponding yaml file stored in the /egs/dataset/config directory, specify the train/test list to use, and run the corresponding bash script
5. To merge models using SA Merge, follow instruction in the script under /merge. Other model merging methods are evaluated using the [mergekit repo](https://github.com/arcee-ai/mergekit)

## Trained Models

coming soon

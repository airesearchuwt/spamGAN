# spamGAN+GPT2

This is the SpamGAN+GPT2 codebase of our paper: *Leveraging GPT-2 for Classifying Spam Reviews with Limited Labeled Data via Adversarial Training*

## Dependencies

Install the dependencies using:

```./install_prereq.sh```

The code was developed with Python 3.6.

## Data

In SpamGAN+GPT2 training, we provide two datasets: Opinion Spam, and Yelp
Reviews. To leverage the most power of GPT-2, we convert the dataset from plain
text to bpe txt files. The example datasets can be viewed in the
[example_datasets](./example_datasets). The `minrun` series is a small subset of
Opinion Spam dataset and the `yelp` series is a small subset of Yelp Reviews
dataset. The full datasets of Opinion Spam and Yelp Reviews can be found in
[final_experiments](./final_experiments).

## Experiment

The full experiment codes can be viewed in
[final_experiments](./final_experiments). If you're interested in reproducing
experiments in the paper, this code may be helpful, albeit requiring some
tweaking. Please note that the full experiment will need a large amount of
training time and at least 32 GBs of GPU memory (in our full experiment, we used
AWS EC2 instance type **g4dn.2xlarge**).

## Adapted GPT-2 

In SpamGAN+GPT2, we adapt the GPT-2 Transformer models from Texar to fit in our
prior work [SpamGAN](https://www.ijcai.org/Proceedings/2019/0723.pdf). The code
of model can be viewed in [custom_texar](./custom_texar). We use the small
version of GPT-2, settings of the small version can be viewed in
[gpt2/gpt2-small](./gpt2/gpt2-small).

## Usage

After installing dependencies, if you want to train with Opinion Spam, simply try:

```python3 spamGAN_train_DCG_gpt2.py spamGAN_config_smallunsup_opspam.json```

to start an experiment with a small subset of Opinion Spam dataset using SpamGAN+GPT2. 

If you want to switch to Yelp Reviews, try:

```python3 spamGAN_train_DCG_gpt2.py spamGAN_config_smallunsup_yelp.json```

to start an experiment with a small subset of Yelp Spam dataset using SpamGAN+GPT2. 

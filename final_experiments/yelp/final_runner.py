import os
import sys
import csv
import json
import random
import time


BASEDIR = '/home/hanfeiyu/Pretrained-spamGAN/final_experiments/yelp/spamGAN_output/'

unsup_revs_path = '/home/hanfeiyu/Pretrained-spamGAN/final_experiments/yelp/spamGAN_output/yelp_unlabeled_reviews_bpe.txt'

train_revs = '/home/hanfeiyu/Pretrained-spamGAN/final_experiments/yelp/spamGAN_output/yelp_train_reviews_bpe.txt'
train_labs = '/home/hanfeiyu/Pretrained-spamGAN/final_experiments/yelp/spamGAN_output/yelp_train_labels.txt'
test_revs = '/home/hanfeiyu/Pretrained-spamGAN/final_experiments/yelp/spamGAN_output/yelp_test_reviews_bpe.txt'
test_labs = '/home/hanfeiyu/Pretrained-spamGAN/final_experiments/yelp/spamGAN_output/yelp_test_labels.txt'


def make_data(trp, usp, run):
    nogan = False
    if usp == -1:
        usp = 0.0
        nogan = True
    with open(train_revs, 'r') as f:
        revs = f.readlines()
    with open(train_labs, 'r') as f:
        labs = f.readlines()
    
    shfl_idx = random.sample(list(range(len(revs))), len(revs))
    revs = [str(revs[i]) for i in shfl_idx]
    labs = [str(labs[i]) for i in shfl_idx]


    total_reviews = revs[:round(trp*len(revs))]
    total_labels = labs[:round(trp*len(revs))]
    tr = total_reviews[:round(0.9*len(total_reviews))]
    tl = total_labels[:round(0.9*len(total_labels))]
    vr = total_reviews[round(0.9*len(total_reviews)):]
    vl = total_labels[round(0.9*len(total_labels)):]
 
 
    with open(unsup_revs_path, 'r') as f:
        unsup_revs_full = f.readlines()
    random.shuffle(unsup_revs_full)
    unsup_revs = unsup_revs_full[:round(usp * len(unsup_revs_full))]

    unsup_labs = ['-1\n'] * len(unsup_revs)


    dir_name = 'tr{}_usp{}_{}'.format(int(trp*100), int(usp * 100), run)
    config_json = 'tr{}_usp{}_config.json'.format(int(trp*100), int(usp * 100))
    
    if nogan:
        dir_name = dir_name + '_nogan/'
        
    os.mkdir(os.path.join(BASEDIR, dir_name))
    curdir = os.path.join(BASEDIR, dir_name)
    ckptdir = os.path.join(BASEDIR, "ckpt")
    
    data_paths = {
        'train_data_reviews' : os.path.join(curdir, 'trevs.txt'),
        'train_data_labels'  : os.path.join(curdir, 'tlabs.txt'),
        'val_data_reviews' : os.path.join(curdir, 'vrevs.txt'),
        'val_data_labels' : os.path.join(curdir, 'vlabs.txt'),
        'unsup_train_data_reviews' : os.path.join(curdir, 'unsup_trevs.txt'),
        'unsup_train_data_labels' : os.path.join(curdir, 'unsup_tlabs.txt'),
        'vocab' : os.path.join(BASEDIR, "gpt2_vocab.txt"),
        'clas_test_ckpts' : [
            "ckpt-all",
            "ckpt-bestclas-acc",
            "ckpt-bestclas-mixed"
            ],
        'clas_pred_output' : "testpreds.txt",
        "gen_perp_output": "perplexities.txt",
        'dir' : curdir,
        "ckptdir": ckptdir,
        "config_json": os.path.join(curdir, config_json)
    }


 
    with open(data_paths['train_data_reviews'], 'w') as f: 
        for x in tr: 
            f.write(x)

    with open(data_paths['train_data_labels'], 'w') as f:
        for x in tl:
            f.write(str(x))
 
    with open(data_paths['unsup_train_data_reviews'], 'w') as f: 
        for x in unsup_revs: 
            f.write(x)
  
    with open(data_paths['unsup_train_data_labels'], 'w') as f:
        for x in unsup_labs:
            f.write(str(x))

    with open(data_paths['val_data_reviews'], 'w') as f:
        for x in vr:
            f.write(x)

    with open(data_paths['val_data_labels'], 'w') as f:
        for x in vl:
            f.write(str(x))


    return data_paths


trp_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
usp_list = [0.0, 0.5, 0.7, 1.0]
iter = 5

for train_pcent in trp_list:
    for unsup_pcent in usp_list:
        for run in range(iter):
            base_config_file = "spamGAN_config_smallunsup_yelp.json"
            base_config = json.loads(open(base_config_file).read())
            data_paths = make_data(train_pcent, unsup_pcent, run)
            
            # inject file paths
            base_config["train_data"]['datasets'][0]['files'] = [data_paths['train_data_reviews'],
                                                              data_paths['unsup_train_data_reviews']]
            base_config["train_data"]['datasets'][1]['files' ] = [data_paths['train_data_labels'],
                                                               data_paths['unsup_train_data_labels']]
                                                               
            base_config["clas_train_data"]['datasets'][0]['files'] = data_paths['train_data_reviews']
            base_config["clas_train_data"]['datasets'][1]['files'] = data_paths['train_data_labels']
            base_config["val_data"]['datasets'][0]['files'] = data_paths['val_data_reviews']
            base_config["val_data"]['datasets'][1]['files'] = data_paths['val_data_labels']
            base_config["test_data"]['datasets'][0]['files'] = test_revs
            base_config["test_data"]['datasets'][1]['files'] = test_labs
            base_config["train_data"]['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config["clas_train_data"]['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config["val_data"]['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config["test_data"]['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config["clas_pred_output"] = data_paths['clas_pred_output']
            base_config["gen_perp_output"] = data_paths['gen_perp_output']
            base_config["log_dir"] = data_paths['dir']
            base_config["checkpoint_dir"] = data_paths['ckptdir']
            print(base_config["train_data"]['datasets'][0]['files'])
            print('Train Pcent {} Unsup Pcent {} Run {}'.format(train_pcent, unsup_pcent, run))
            
            # Run
            with open(data_paths["config_json"], "w") as train_config:
                json.dump(base_config, train_config)
            
            train_status = 114514
            while train_status is not 0:
                train_status = os.system("nice -n 10 python3 spamGAN_train_DCG_gpt2.py {}".format(data_paths["config_json"]))
            
            # Unit test
            base_config["gen_clas_test"] = True
            
            for ckpt in data_paths['clas_test_ckpts']:
                base_config["clas_test_ckpts"] = ckpt
                with open(data_paths["config_json"], "w") as test_config:
                    json.dump(base_config, test_config)
                    
                test_status = 114514
                while test_status is not 0:
                    test_status = os.system("nice -n 10 python3 spamGAN_train_DCG_gpt2.py {}".format(data_paths["config_json"])) 
            
            # Clean disk space
            os.system("rm ./spamGAN_output/ckpt/*")  


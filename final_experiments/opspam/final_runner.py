import os
import sys
import csv
import importlib
import spamGAN_train_DCG_gpt2
import spamGAN_train_DCG_gpt2_cpu
import random
import time


BASEDIR = '/home/hanfeiyu/Pretrained-spamGAN/final_experiments/opspam/spamGAN_output/'

unsup_revs_path = '/home/hanfeiyu/Pretrained-spamGAN/final_experiments/opspam/spamGAN_output/chicago_unlab_reviews_bpe.txt'

train_revs = '/home/hanfeiyu/Pretrained-spamGAN/final_experiments/opspam/spamGAN_output/opspam_train_reviews_bpe.txt'
train_labs = '/home/hanfeiyu/Pretrained-spamGAN/final_experiments/opspam/spamGAN_output/opspam_train_labels.txt'
test_revs = '/home/hanfeiyu/Pretrained-spamGAN/final_experiments/opspam/spamGAN_output/opspam_test_reviews_bpe.txt'
test_labs = '/home/hanfeiyu/Pretrained-spamGAN/final_experiments/opspam/spamGAN_output/opspam_test_labels.txt'


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


    tr = revs[:round(trp *len(revs))]
    vr = revs[round(trp * len(revs)):]
    tl = labs[:round(trp * len(revs))]
    vl = labs[round(trp * len(revs)):]
 
    if len(vr) == 0 :
        # just add a fake as a workaround
        vr = revs[0:100]
        vl = labs[0:100]
    with open(unsup_revs_path, 'r') as f:
        unsup_revs_full = f.readlines()
    random.shuffle(unsup_revs_full)
    unsup_revs = unsup_revs_full[:round(usp * len(unsup_revs_full))]

    unsup_labs = ['-1\n'] * len(unsup_revs)


    dir_name = 'tr{}_usp{}_{}'.format(int(trp*100), int(usp * 100), run)
    time_result_file = 'tr{}_usp{}_time'.format(int(trp*100), int(usp * 100))
    all_result_file = 'tr{}_usp{}_all'.format(int(trp*100), int(usp * 100))
    bestclas_pretrain_result_file = 'tr{}_usp{}_bestclas_pretrain'.format(int(trp*100), int(usp * 100))
    bestclas_acc_result_file = 'tr{}_usp{}_bestclas_acc'.format(int(trp*100), int(usp * 100))
    bestclas_f1_result_file = 'tr{}_usp{}_bestclas_f1'.format(int(trp*100), int(usp * 100))
    bestclas_mixed_result_file = 'tr{}_usp{}_bestclas_mixed'.format(int(trp*100), int(usp * 100))
    
    if nogan:
        dir_name = dir_name + '_nogan/'
        
    os.mkdir(os.path.join(BASEDIR, dir_name))
    curdir = os.path.join(BASEDIR, dir_name)
    resultdir = os.path.join(BASEDIR, "result")
    time_result_file = os.path.join(resultdir, time_result_file)
    all_result_file = os.path.join(resultdir, all_result_file)
    bestclas_pretrain_result_file = os.path.join(resultdir, bestclas_pretrain_result_file)
    bestclas_acc_result_file = os.path.join(resultdir, bestclas_acc_result_file)
    bestclas_f1_result_file = os.path.join(resultdir, bestclas_f1_result_file)
    bestclas_mixed_result_file = os.path.join(resultdir, bestclas_mixed_result_file)
    
    data_paths = {
        'train_data_reviews' : os.path.join(curdir, 'trevs.txt'),
        'train_data_labels'  : os.path.join(curdir, 'tlabs.txt'),
        'val_data_reviews' : os.path.join(curdir, 'vrevs.txt'),
        'val_data_labels' : os.path.join(curdir, 'vlabs.txt'),
        'unsup_train_data_reviews' : os.path.join(curdir, 'unsup_trevs.txt'),
        'unsup_train_data_labels' : os.path.join(curdir, 'unsup_tlabs.txt'),
        'vocab' : "/home/hanfeiyu/Pretrained-spamGAN/final_experiments/opspam/spamGAN_output/gpt2_vocab.txt",
        'clas_test_ckpts' : [os.path.join(curdir, "ckpt-all"),
                             os.path.join(curdir, 'ckpt-bestclas-pretrain'),
                             os.path.join(curdir, "ckpt-bestclas-acc"),
                             os.path.join(curdir, "ckpt-bestclas-f1"),
                             os.path.join(curdir, "ckpt-bestclas-mixed")],
        'clas_pred_output' : os.path.join(curdir, 'testpreds.txt'),
        "gen_perp_output": os.path.join(curdir, "perplexities.txt"),
        'dir' : curdir,
        'time_result_file' : time_result_file,
        'all_result_file' : all_result_file,
        'bestclas_pretrain_result_file' : bestclas_pretrain_result_file,
        'bestclas_acc_result_file' : bestclas_acc_result_file,
        'bestclas_f1_result_file' : bestclas_f1_result_file,
        'bestclas_mixed_result_file' : bestclas_mixed_result_file,
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


# 0.5, 0.8 x 0.5, 0.8
for train_pcent in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    for unsup_pcent in [0.0, 0.5, 0.7, 1.0]:
        for run in range(10):
            base_config_file = 'spamGAN_config_smallunsup_opspam'
            data_paths = make_data(train_pcent, unsup_pcent, run)
            importlib.invalidate_caches()
            base_config = importlib.import_module(base_config_file)
            base_config = importlib.reload(base_config)
            # inject file paths
            base_config.train_data['datasets'][0]['files'] = [data_paths['train_data_reviews'],
                                                              data_paths['unsup_train_data_reviews']]
            base_config.train_data['datasets'][1]['files' ] = [data_paths['train_data_labels'],
                                                               data_paths['unsup_train_data_labels']]
                                                               
            base_config.clas_train_data['datasets'][0]['files'] = data_paths['train_data_reviews']
            base_config.clas_train_data['datasets'][1]['files'] = data_paths['train_data_labels']
            base_config.val_data['datasets'][0]['files'] = data_paths['val_data_reviews']
            base_config.val_data['datasets'][1]['files'] = data_paths['val_data_labels']
            base_config.test_data['datasets'][0]['files'] = test_revs
            base_config.test_data['datasets'][1]['files'] = test_labs
            base_config.train_data['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config.clas_train_data['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config.val_data['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config.test_data['datasets'][0]['vocab_file'] = data_paths['vocab']
            base_config.clas_test_ckpts = data_paths['clas_test_ckpts']
            base_config.clas_pred_output = data_paths['clas_pred_output']
            base_config.gen_perp_output = data_paths['gen_perp_output']
            base_config.log_dir = data_paths['dir']
            base_config.checkpoint_dir = data_paths['dir']
            print(base_config.train_data['datasets'][0]['files'])
            print('Train Pcent {} Unsup Pcent {} Run {}'.format(train_pcent, unsup_pcent, run))
            
            # Run
            dict_time_res = spamGAN_train_DCG_gpt2.main(base_config)
            
            # Record time metrics
            time_file_exists = os.path.isfile(data_paths["time_result_file"])
            f = open(data_paths["time_result_file"],'a')
            w = csv.DictWriter(f, dict_time_res.keys())
            if not time_file_exists:
                print("Writing time header...")
                w.writeheader()
            w.writerow(dict_time_res)
            f.close()
            
            # Unit test
            base_config.gen_clas_test = True
            dict_all_res, dict_bestclas_pretrain_res, dict_bestclas_acc_res, dict_bestclas_f1_res, dict_bestclas_mixed_res = spamGAN_train_DCG_gpt2_cpu.main(base_config)

            all_file_exists = os.path.isfile(data_paths["all_result_file"])
            f = open(data_paths["all_result_file"],'a')
            w = csv.DictWriter(f, dict_all_res.keys())
            if not all_file_exists:
                print("Writing all header...")
                w.writeheader()
            w.writerow(dict_all_res)
            f.close()
            
            bestclas_pretrain_file_exists = os.path.isfile(data_paths["bestclas_pretrain_result_file"])
            f = open(data_paths["bestclas_pretrain_result_file"],'a')
            w = csv.DictWriter(f, dict_bestclas_pretrain_res.keys())
            if not bestclas_pretrain_file_exists:
                print("Writing bestclas-pretrain header...")
                w.writeheader()
            w.writerow(dict_bestclas_pretrain_res)
            f.close()
            
            bestclas_acc_file_exists = os.path.isfile(data_paths["bestclas_acc_result_file"])
            f = open(data_paths["bestclas_acc_result_file"],'a')
            w = csv.DictWriter(f, dict_bestclas_acc_res.keys())
            if not bestclas_acc_file_exists:
                print("Writing bestclas-acc header...")
                w.writeheader()
            w.writerow(dict_bestclas_acc_res)
            f.close()
            
            bestclas_f1_file_exists = os.path.isfile(data_paths["bestclas_f1_result_file"])
            f = open(data_paths["bestclas_f1_result_file"],'a')
            w = csv.DictWriter(f, dict_bestclas_f1_res.keys())
            if not bestclas_f1_file_exists:
                print("Writing bestclas-f1 header...")
                w.writeheader()
            w.writerow(dict_bestclas_f1_res)
            f.close()
            
            bestclas_mixed_file_exists = os.path.isfile(data_paths["bestclas_mixed_result_file"])
            f = open(data_paths["bestclas_mixed_result_file"],'a')
            w = csv.DictWriter(f, dict_bestclas_mixed_res.keys())
            if not bestclas_mixed_file_exists:
                print("Writing bestclas-mixed header...")
                w.writeheader()
            w.writerow(dict_bestclas_mixed_res)
            f.close()
            




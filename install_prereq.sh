#! /bin/bash

eg_ckpt_dir="./ckpt"
final_outputs="./final_experiments/opspam/spamGAN_output ./final_experiments/yelp/spamGAN_output"

if [ ! -d "$eg_ckpt_dir" ]
then
    mkdir $eg_ckpt_dir
else
    rm $eg_ckpt_dir/*
fi

for output in $final_outputs
do
    if [ ! -d "$output/ckpt" ]
    then
        mkdir $output/ckpt
    else
        rm $output/ckpt/*
    fi

    if [ ! -d "$output/result" ]
    then
        mkdir $output/result
    # else
    #     rm $output/ckpt/*
    fi
done

pip3 install tensorflow-probability==0.7.0 texar==0.2.4
        

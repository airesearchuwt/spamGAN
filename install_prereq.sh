#! /bin/bash

eg_ckpt_dir="./ckpt"

if [ ! -d "$eg_ckpt_dir" ]
then
    mkdir $eg_ckpt_dir
else
    rm $eg_ckpt_dir/*
fi

pip3 install tensorflow-probability==0.7.0 texar==0.2.4
        

#! /bin/bash

ckpt_dir="ckpt"

if [ ! -d "$ckpt_dir" ]
then
    mkdir $ckpt_dir
else
    rm $ckpt_dir/*
fi

pip3 install tensorflow-probability==0.7.0 texar==0.2.4
        

#! /bin/bash

output_path="spamGAN_output"

ps aux | grep -v grep | grep final | awk {'print $2'} | xargs kill
rm ./$output_path/ckpt/*
#rm ./$output_path/result/*

experiment_folders=`ls -l $output_path | grep usp | awk {'print $9'}`

for folder in $experiment_folders
do
    rm -rf ./$output_path/$folder
done

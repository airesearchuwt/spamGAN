#! /bin/bash

#
# Activate tensorflow_p36 permanently on g4dn.2xlarge
#

source activate tensorflow_p36

echo "" >> ~/.bashrc
echo "# Activate tensorflow_p36" >> ~/.bashrc
echo "source activate tensorflow_p36" >> ~/.bashrc
echo "" >> ~/.bashrc

#
# Create ckpt and result folders if not exist
#

final_outputs="./opspam/spamGAN_output ./yelp/spamGAN_output"

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
    #     rm $output/result/*
    fi
done

#
# Install necessary dependencies
#

pip3 install tensorflow-probability==0.7.0 texar==0.2.4
        

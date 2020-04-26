#! /bin/bash

ps aux | grep spam | grep -v grep | awk {'print $2'} | xargs kill
find /tmp/ -size +10000k | xargs rm
find ./ckpt/* | xargs rm

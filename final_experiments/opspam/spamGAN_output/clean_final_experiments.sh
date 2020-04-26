#! /bin/bash

#ps aux | grep -v grep | grep final | awk {'print $2'} | xargs kill
sudo rm -rf `ls -l | grep usp | awk {'print $9'}`
rm result/*

#!/bin/bash

#echo "Copying imagenet.tar to /tmp/"
#cp /scratch/05714/jgpaul/imagenet.tar /tmp/
#echo "Done copying. Extracting"
#tar xf /tmp/imagenet.tar -C /tmp/ --strip-components 5
echo "Copying imagenet to /tmp"
cp -r /scratch/05714/jgpaul/imagenet/* /tmp/
echo "Done"

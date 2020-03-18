#!/bin/bash

rm -rf /tmp/ILSVRC2012_img_train
rm -rf /tmp/ILSVRC2012_img_val

echo "Copying imagenet.tar to /tmp/"
cp /scratch/05714/jgpaul/imagenet.tar /tmp/
echo "Done copying. Extracting"
tar xf /tmp/imagenet.tar -C /tmp/
echo "Copying imagenet to /tmp"

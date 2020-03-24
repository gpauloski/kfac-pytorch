#!/bin/bash

echo "Copying imagenet.tar to /tmp/"
cp /scratch/05714/jgpaul/imagenet.tar /tmp/
echo "Done copying. Extracting"
tar xf /tmp/imagenet.tar -C /tmp/
echo "Done."

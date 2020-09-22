#!/bin/bash

# Copy imagenet tar file to /tmp/ and extract.
# Use --tiny to copy a smaller version of imagenet instead.
# Produces directories: /tmp/imagenet/ILSVRC2012_img_{train,val}

tiny=${tiny:-false}

while [ $# -gt 0 ]; do
  if [[ $1 == *"--"* ]]; then
      param="${1/--/}"
      declare $param=true
  fi
  shift
done

DEST=/tmp
if [ "$tiny" == true ]; then
  SOURCE=/scratch/05714/jgpaul/datasets/imagenet-tiny.tar
  STRIP=1
else
  SOURCE=/scratch/05714/jgpaul/datasets/imagenet.tar
  STRIP=5
fi

echo Copying $SOURCE to $DEST
cp $SOURCE $DEST/imagenet.tar
echo "Done copying. Extracting"
mkdir $DEST/imagenet
tar xf $DEST/imagenet.tar -C $DEST/imagenet --strip-components $STRIP
echo "Done."

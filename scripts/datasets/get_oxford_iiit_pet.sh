#!/bin/bash

set -evx


PROJECT_ROOT=$(cd "$(dirname $0)/../.."; pwd)

data_path=$PROJECT_ROOT/data/oxford-pet/

if [ ! -d "$data_path" ]; then
    mkdir -p "$data_path"
fi


if [ ! -f "$data_path/saint_bernard/saint_bernard_33.jpg" ]; then

pushd $data_path

# Downloading the dataset
wget https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz
tar zxvf oxford-iiit-pet.tgz
rm oxford-iiit-pet.tgz
mv oxford-iiit-pet/images/* .
rm -rf oxford-iiit-pet
rm *.mat

# Organizing images into folders
for image in *jpg ; do
  label=`echo $image | awk -F_ '{gsub($NF,"");sub(".$", "");print}'`
  mkdir -p $label
  mv $image $label/$image
done

popd

fi


# Making .lst and .rec files for MXNet to load
if [ ! -f "$data_path/data_train2.lst" ]; then

# Cleaning up the images that are failing with OpenCV
rm -f $data_path/Abyssinian/Abyssinian_34.jpg
rm -f $data_path/Egyptian_Mau/Egyptian_Mau_139.jpg
rm -f $data_path/Egyptian_Mau/Egyptian_Mau_145.jpg
rm -f $data_path/Egyptian_Mau/Egyptian_Mau_167.jpg
rm -f $data_path/Egyptian_Mau/Egyptian_Mau_177.jpg
rm -f $data_path/Egyptian_Mau/Egyptian_Mau_191.jpg

python $MXNET_HOME/tools/im2rec.py \
  --list \
  --train-ratio 0.8 \
  --recursive \
  $data_path/data $data_path

python $MXNET_HOME/tools/im2rec.py \
  --resize 224 \
  --center-crop \
  --num-thread 4 \
  $data_path/data $data_path

fi

#!/bin/bash

set -evx

mkdir -p model
cd model

wget http://data.mxnet.io/models/imagenet/inception-bn/Inception-BN-symbol.json
wget http://data.mxnet.io/models/imagenet/inception-bn/Inception-BN-0126.params
mv Inception-BN-0126.params Inception-BN-0000.params

# Downloading ImageNet Categories
wget http://data.mxnet.io/models/imagenet/synset.txt

cd ...

#!/bin/bash

set -evx

mkdir -p model
cd model
wget http://data.mxnet.io/models/imagenet/vgg/vgg16-symbol.json
wget http://data.mxnet.io/models/imagenet/vgg/vgg16-0000.params
cd ..

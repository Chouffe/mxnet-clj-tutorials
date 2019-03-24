#!/bin/bash

set -evx

mkdir -p model
cd model

wget http://data.mxnet.io/models/imagenet/resnet/152-layers/resnet-152-symbol.json
wget http://data.mxnet.io/models/imagenet/resnet/152-layers/resnet-152-0000.params

cd ..

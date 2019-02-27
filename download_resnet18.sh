#!/bin/bash

set -evx

mkdir -p model
cd model
wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-symbol.json
wget http://data.mxnet.io/models/imagenet/resnet/18-layers/resnet-18-0000.params
cd ..

#!/bin/bash

set -evx

PROJECT_ROOT=$(cd "$(dirname $0)/../.."; pwd)

model_path=$PROJECT_ROOT/model/

if [ ! -f "$model_path/Inception-BN-0000.params" ]; then
  pushd $model_path
  wget http://data.mxnet.io/models/imagenet/inception-bn/Inception-BN-symbol.json
  wget http://data.mxnet.io/models/imagenet/inception-bn/Inception-BN-0126.params
mv Inception-BN-0126.params Inception-BN-0000.params
# Downloading ImageNet Categories
wget http://data.mxnet.io/models/imagenet/synset.txt

  popd
fi

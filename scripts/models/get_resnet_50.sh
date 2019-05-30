#!/bin/bash

set -evx

PROJECT_ROOT=$(cd "$(dirname $0)/../.."; pwd)

model_path=$PROJECT_ROOT/model/

if [ ! -f "$model_path/resnet-50-0000.params" ]; then
  wget http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50-symbol.json -P $model_path
  wget http://data.mxnet.io/models/imagenet/resnet/50-layers/resnet-50-0000.params -P $model_path
  wget http://data.mxnet.io/models/imagenet/resnet/synset.txt -P $model_path
fi


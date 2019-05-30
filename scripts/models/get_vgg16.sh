#!/bin/bash

set -evx

PROJECT_ROOT=$(cd "$(dirname $0)/../.."; pwd)

model_path=$PROJECT_ROOT/model/

if [ ! -f "$model_path/vgg16-0000.params" ]; then
  wget http://data.mxnet.io/models/imagenet/vgg/vgg16-symbol.json -P $model_path
  wget http://data.mxnet.io/models/imagenet/vgg/vgg16-0000.params -P $model_path
fi

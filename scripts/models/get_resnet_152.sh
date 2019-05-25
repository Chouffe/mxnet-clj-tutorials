#!/bin/bash

set -evx

PROJECT_ROOT=$(cd "$(dirname $0)/../.."; pwd)

model_path=$PROJECT_ROOT/model/


if [ ! -f "$model_path/resnet-152-0000.params" ]; then
  wget http://data.mxnet.io/models/imagenet/resnet/152-layers/resnet-152-symbol.json
  wget http://data.mxnet.io/models/imagenet/resnet/152-layers/resnet-152-0000.params
fi


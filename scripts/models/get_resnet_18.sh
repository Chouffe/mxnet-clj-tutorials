#!/bin/bash

set -evx

PROJECT_ROOT=$(cd "$(dirname $0)/../.."; pwd)

model_path=$PROJECT_ROOT/model/

if [ ! -f "$model_path/resnet-18-0000.params" ]; then
  wget https://s3.us-east-2.amazonaws.com/scala-infer-models/resnet-18/resnet-18-symbol.json -P $model_path
  wget https://s3.us-east-2.amazonaws.com/scala-infer-models/resnet-18/resnet-18-0000.params -P $model_path
  wget https://s3.us-east-2.amazonaws.com/scala-infer-models/resnet-18/synset.txt -P $model_path
fi

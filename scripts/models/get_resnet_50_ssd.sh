#!/bin/bash

set -evx

PROJECT_ROOT=$(cd "$(dirname $0)/../.."; pwd)

model_path=$PROJECT_ROOT/model/resnet50_ssd/

image_path=$PROJECT_ROOT/data/resnet50_ssd/

if [ ! -d "$model_path" ]; then
  mkdir -p "$model_path"
fi

if [ ! -d "$image_path" ]; then
  mkdir -p "$image_path"
fi

if [ ! -f "$model_path/resnet50_ssd_model-0000.params" ]; then
  wget https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-symbol.json -P $model_path
  wget https://s3.amazonaws.com/model-server/models/resnet50_ssd/resnet50_ssd_model-0000.params -P $model_path
  wget https://s3.amazonaws.com/model-server/models/resnet50_ssd/synset.txt -P $model_path
fi

if [ ! -f "$image_path/000001.jpg" ]; then
    cd $image_path
    wget https://cloud.githubusercontent.com/assets/3307514/20012566/cbb53c76-a27d-11e6-9aaa-91939c9a1cd5.jpg -O 000001.jpg
    wget https://cloud.githubusercontent.com/assets/3307514/20012567/cbb60336-a27d-11e6-93ff-cbc3f09f5c9e.jpg -O dog.jpg
    wget https://cloud.githubusercontent.com/assets/3307514/20012563/cbb41382-a27d-11e6-92a9-18dab4fd1ad3.jpg -O person.jpg
fi

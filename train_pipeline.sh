#!/bin/sh

SCRIPT_PATH=$(dirname "$(realpath "$0")")
KALPA_PATH=$SCRIPT_PATH
TRAINING_PATH=$SCRIPT_PATH/ai8x_training
DATASET_NAME=top_viewed_people

cp $KALPA_PATH/datasets/* $TRAINING_PATH/datasets/

# Start training
cd $TRAINING_PATH
. bin/activate
python3 train.py --epochs 20 --data $KALPA_PATH/data_from_dvc --dataset $DATASET_NAME --model ai85ressimplenet --compress $KALPA_PATH/schedule-kalpa.yaml --optimizer Adam --lr 0.001 --deterministic --confusion --param-hist --embedding --device MAX78000 --cpu --out-dir $KALPA_PATH/net --qat-policy $KALPA_PATH/qat_kalpa_policy.yaml --mlflow-uri=http://172.16.0.2:5000
deactivate

#!/bin/sh

SCRIPT_PATH=$(dirname "$(realpath "$0")")
KALPA_PATH=$SCRIPT_PATH
TRAINING_PATH=$SCRIPT_PATH/ai8x_training
DATASET_NAME=top_viewed_people

# Sync user data and datasets
# TODO esiste un modo migliore senza fare la copia?
rsync -avx $KALPA_PATH/datasets/ $TRAINING_PATH/datasets/
rsync -avx $KALPA_PATH/models/ $TRAINING_PATH/models/

# Start training
cd $TRAINING_PATH
. bin/activate
python3 train.py --epochs 100 --data $KALPA_PATH/data --dataset $DATASET_NAME --model ai85ressimplenet --compress $KALPA_PATH/schedule-kalpa.yaml --optimizer Adam --lr 0.001 --deterministic --confusion --param-hist --embedding --device MAX78000 --cpu --enable-tensorboard --out-dir $KALPA_PATH/net --qat-policy $KALPA_PATH/qat_kalpa_policy.yaml
deactivate

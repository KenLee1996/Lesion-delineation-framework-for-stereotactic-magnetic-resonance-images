#!/usr/bin/env bash

my_dir="$(dirname "$0")"
. $my_dir/set_env.sh

echo "MMAR_ROOT set to $MMAR_ROOT"

# Data list containing all data
CONFIG_FILE=config/config_train_0130.json
CLIENT_FILE=config/config_fed_client.json
ENVIRONMENT_FILE=config/environment.json

python3 -u  -m nvmidl.apps.fed_learn.client.fed_local_train \
    -m $MMAR_ROOT \
    -c $CONFIG_FILE \
    -e $ENVIRONMENT_FILE \
    -s $CLIENT_FILE \
    --set \
    DATASET_JSON=$MMAR_ROOT/config/brain_tumor1.json \
    secure_train=false \
    uid=client1

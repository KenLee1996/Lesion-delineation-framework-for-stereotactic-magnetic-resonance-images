#!/usr/bin/env bash

my_dir="$(dirname "$0")"
. $my_dir/set_env.sh

echo "MMAR_ROOT set to $MMAR_ROOT"

export CUDA_VISIBLE_DEVICES=

# Data list containing all data
CONFIG_FILE=config/config_train_0826.json
SERVER_FILE=config/config_fed_server.json
ENVIRONMENT_FILE=config/environment.json

python3 -u  -m nvmidl.apps.fed_learn.server.fed_aggregate \
    -m $MMAR_ROOT \
    -c $CONFIG_FILE \
    -e $ENVIRONMENT_FILE \
    -s $SERVER_FILE \
    --set \    
    secure_train=false



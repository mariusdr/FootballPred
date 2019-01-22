#!/bin/sh

PROJECT_ROOT=~/Projects/FootballPred

# saving model weights etc.
SAVE_PATH=$PROJECT_ROOT/saved/$(date +%d.%m_%H:%M)
PATH_SAVED_LOGS=$SAVE_PATH/logs
PATH_SAVED_MODELS=$SAVE_PATH/models
PATH_SAVED_STATS=$SAVE_PATH/stats

mkdir $SAVE_PATH
cd $SAVE_PATH
mkdir logs
mkdir models
mkdir stats
cd $PROJECT_ROOT

PATH_LOGFILE=$PATH_SAVED_LOGS/train_log.txt

# system config
PATH_DB=$PROJECT_ROOT/data/database.sqlite

DEVICE="cpu"
BIG_DATASET=0
LOGLEVEL="INFO"

# hyperparameters for model
HIDDEN_SIZE=128
HIDDEN_LAYERS=4
BIDIRECTIONAL=0

# hyperparameters for training
EPOCHS=250
BATCH_SIZE=8

OPTIMIZER="Adam"
LEARNING_RATE=0.001
WEIGHT_DECAY=0.0

#OPTIMIZER="SGD"
#LEARNING_RATE=0.01
MOMENTUM=0.01

# run
EXEC=$PROJECT_ROOT/src/main.py

python3 $EXEC --database $PATH_DB --log $PATH_LOGFILE --epochs $EPOCHS --batch_size $BATCH_SIZE --model_save_path $PATH_SAVED_MODELS --stats_save_path $PATH_SAVED_STATS --optimizer $OPTIMIZER --learning_rate $LEARNING_RATE --weight_decay $WEIGHT_DECAY --lstm_hidden_size $HIDDEN_SIZE --lstm_hidden_layers $HIDDEN_LAYERS --bidirectional $BIDIRECTIONAL --momentum $MOMENTUM --big_dataset $BIG_DATASET


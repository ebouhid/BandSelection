#!/bin/bash

nohup python train.py --loss gbcloss
nohup python train.py --loss bce
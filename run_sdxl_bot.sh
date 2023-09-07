#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate sdxlbot

python3 sdxl_bot.py "$@"

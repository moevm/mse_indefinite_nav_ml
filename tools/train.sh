#!/bin/bash
Xvfb :0 -screen 1 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &
export DISPLAY=:1


python3 ./tools/train.py --conf_path ./configs/conf.py
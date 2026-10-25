#!/usr/bin/env bash
# Recorder for qual-cheese-salami-final-model: each episode gets topping + quadrant labels.
cd "$(dirname "$0")"
exec .venv/bin/python recorder.py \
  --root ../../data/qual-cheese-salami-final-model \
  --dataset-name qual-cheese-salami-final-model \
  --repo yaditya/qual-cheese-salami-final-model \
  --camera-order 1,0 --fps 10 --image-size 224 --port 8891 \
  --task "pick-n-put-cheese put-in-q1" \
  --categories \
    "pick-n-put-cheese put-in-q1" "pick-n-put-cheese put-in-q2" \
    "pick-n-put-cheese put-in-q3" "pick-n-put-cheese put-in-q4" \
    "pick-n-put-salami put-in-q1" "pick-n-put-salami put-in-q2" \
    "pick-n-put-salami put-in-q3" "pick-n-put-salami put-in-q4" \
  "$@"

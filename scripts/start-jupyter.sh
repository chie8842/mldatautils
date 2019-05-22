#!/bin/bash

nohup jupyter lab --ip=0.0.0.0 --port=${PORT} --no-browser --NotebookApp.token='test' --NotebookApp.iopub_data_rate_limit=10000000000 --allow-root &


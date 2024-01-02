#!/bin/bash
#SBATCH --job-name=train_multiple
#SBATCH --output=train_multiple_output.txt

# train RR
python main.py --mode train --model RCNN --save_dir train_models/RCNN_2000_drop0.5_2 --log_filename train_record.txt

python main.py --mode train --model RCNN --save_dir train_models/RCNN_2000_drop0.5_3 --log_filename train_record.txt


#!/bin/bash
#SBATCH --job-name=train_multiple
#SBATCH --output=test_multiple_output.txt


python main.py --mode test --model RCNN --load_path train_models/RCNN_2000_drop0.5_2/best_model --save_dir train_models/RCNN_2000_drop0.5_2 --log_filename test_record.txt
python main.py --mode test --model RCNN --load_path train_models/RCNN_2000_drop0.5_3/best_model --save_dir train_models/RCNN_2000_drop0.5_3 --log_filename test_record.txt

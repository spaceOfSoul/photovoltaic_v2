#!/bin/bash
#SBATCH --job-name=train_multiple
#SBATCH --output=train_multiple_output.txt

# train LL
python main.py --mode train --model correction_LSTMs --save_dir train_models/2-stageLL_2000_drop0.5_2 --log_filename train_record.txt

python main.py --mode train --model correction_LSTMs --save_dir train_models/2-stageLL_2000_drop0.5_3 --log_filename train_record.txt


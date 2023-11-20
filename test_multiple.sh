#!/bin/bash
#SBATCH --job-name=train_multiple
#SBATCH --output=train_multiple_output.txt

# train RR
python main.py --mode train --model 2-stageRR --save_dir train_models/2-stageRR_2000_drop0.5_2 --log_filename train_record.txt

python main.py --mode train --model 2-stageRR --save_dir train_models/2-stageRR_2000_drop0.5_3 --log_filename train_record.txt

# train RL
python main.py --mode train --model 2-stageRL --save_dir train_models/2-stageRL_2000_drop0.5_2 --log_filename train_record.txt

python main.py --mode train --model 2-stageRL --save_dir train_models/2-stageRL_2000_drop0.5_3 --log_filename train_record.txt

# train LR
python main.py --mode train --model 2-stageLR --save_dir train_models/2-stageLR_2000_drop0.5_2 --log_filename train_record.txt

python main.py --mode train --model 2-stageLR --save_dir train_models/2-stageLR_2000_drop0.5_3 --log_filename train_record.txt

# train LL
python main.py --mode train --model 2-stageLL --save_dir train_models/2-stageLL_2000_drop0.5_2 --log_filename train_record.txt

python main.py --mode train --model 2-stageLL --save_dir train_models/2-stageLL_2000_drop0.5_3 --log_filename train_record.txt

# train RNN
python main.py --mode train --model RNN --save_dir train_models/RNN_2000_drop0.5_2 --log_filename train_record.txt

python main.py --mode train --model RNN --save_dir train_models/RNN_2000_drop0.5_3 --log_filename train_record.txt

# train LSTM
python main.py --mode train --model LSTM --save_dir train_models/LSTM_2000_drop0.5_2 --log_filename train_record.txt

python main.py --mode train --model LSTM --save_dir train_models/LSTM_2000_drop0.5_3 --log_filename train_record.txt

#!/bin/bash

#SBATCH -J PVF
#SBATCH -o training_%j.out
#SBATCH --nodes=1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pv_forecasting
python main.py --mode train --save_dir ./Models/RNN1_sL60_loc104/ --seqLeng 60 --loc_ID 104
python main.py --mode train --save_dir ./Models/RNN1_sL60_loc105/ --seqLeng 60 --loc_ID 105
python main.py --mode train --save_dir ./Models/RNN1_sL60_loc106/ --seqLeng 60 --loc_ID 106
python main.py --mode train --save_dir ./Models/RNN1_sL60_loc523/ --seqLeng 60 --loc_ID 523
python main.py --mode train --save_dir ./Models/RNN1_sL60_loc524/ --seqLeng 60 --loc_ID 524
python main.py --mode train --save_dir ./Models/RNN1_sL60_loc566/ --seqLeng 60 --loc_ID 566
python main.py --mode train --save_dir ./Models/RNN1_sL60_loc580/ --seqLeng 60 --loc_ID 580
python main.py --mode train --save_dir ./Models/RNN1_sL60_loc678/ --seqLeng 60 --loc_ID 678




#python main.py --mode test --load_dir ./test/best_model --test_solar_dir ../dataset/photovoltaic/GWNU_C3/

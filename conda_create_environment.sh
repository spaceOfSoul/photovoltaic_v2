conda create -y -b pv_forecasting python=3.8.5
conda activate pv_forecasting

conda install -y numba
conda install -y pandas h5py scipy
conda install -y pytorch torchvision cudatoolkit -c pytorch

pip install -r requirements.txt

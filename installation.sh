export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:
export PATH=$PATH:$CUDA_HOME/bin

pip install -r requirements.txt
cd pointnet2
export MAX_JOB=1
sudo python3 setup.py install
cd ../knn
sudo python3 setup.py install
cd ../graspnetAPI
pip install .
cd ..

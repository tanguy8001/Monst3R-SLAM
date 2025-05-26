source /work/courses/3dv/24/24_envs/bin/activate
which conda

python main_monster_slam.py --dataset datasets/tum2/rgbd_dataset_freiburg3_walking_xyz/ --config config/calib.yaml

conda create -n 3dv python=3.11 cmake=3.14.0
conda activate 3dv
# CUDA 11.8
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1  pytorch-cuda=11.8 -c pytorch -c nvidia

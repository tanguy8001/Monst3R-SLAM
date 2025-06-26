#!/bin/bash
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --time=720
#SBATCH --output=out/run_main_monster_slam_%j.out

cd /work/courses/3dv/24/MASt3R-SLAM
source /work/courses/3dv/24/24_envs/bin/activate
conda activate 3dv

echo "Starting main_monster_slam at: $(date)"

#python main_monster_slam.py --dataset datasets/tum2/rgbd_dataset_freiburg3_walking_xyz/ --config config/calib.yaml
python main_monster_slam.py --dataset datasets/bonn/rgbd_bonn_person_tracking/ --config config/calib.yaml
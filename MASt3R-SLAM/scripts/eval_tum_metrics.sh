#!/bin/bash
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --time=720
#SBATCH --output=eval_tum_metrics%j.out

cd /work/courses/3dv/24/MASt3R-SLAM
source /home/tdieudonne/.bashrc
conda activate slam

echo "Starting metrics computation of TUM dataset at: $(date)"

dataset_path="datasets/tum/"
datasets=(
    rgbd_dataset_freiburg1_360
    rgbd_dataset_freiburg1_desk
    rgbd_dataset_freiburg1_desk2
    rgbd_dataset_freiburg1_floor
    rgbd_dataset_freiburg1_plant
    rgbd_dataset_freiburg1_room
    rgbd_dataset_freiburg1_rpy
    rgbd_dataset_freiburg1_teddy
    rgbd_dataset_freiburg1_xyz
)

no_calib=false
print_only=false
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --no-calib)
            no_calib=true
            ;;
        --print)
            print_only=true
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

for dataset in ${datasets[@]}; do
    dataset_name="$dataset_path""$dataset"/
    echo ${dataset_name}
    if [ "$no_calib" = true ]; then
        evo_ape tum $dataset_name/groundtruth.txt logs/tum/no_calib/$dataset/$dataset.txt -as
    else
        evo_ape tum $dataset_name/groundtruth.txt logs/tum/calib/$dataset/$dataset.txt -as
    fi
done

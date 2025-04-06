#!/bin/bash
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --time=720
#SBATCH --output=out/eval_tum2_metrics_%j.out

cd /work/courses/3dv/24/MASt3R-SLAM
source /home/tdieudonne/.bashrc
conda activate slam

echo "Starting evaluation of TUM 2 dataset at: $(date)"

dataset_path="datasets/tum2/"
datasets=(
    rgbd_dataset_freiburg2_desk_with_person
    rgbd_dataset_freiburg3_sitting_halfsphere
    rgbd_dataset_freiburg3_sitting_rpy
    rgbd_dataset_freiburg3_sitting_static
    rgbd_dataset_freiburg3_sitting_xyz
    rgbd_dataset_freiburg3_walking_halfsphere  # Out of memory
    rgbd_dataset_freiburg3_walking_rpy  # Out of memory
    rgbd_dataset_freiburg3_walking_static
    rgbd_dataset_freiburg3_walking_xyz
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
    dataset_name="$dataset_path""$dataset"
    echo ${dataset_name}
    if [ "$no_calib" = true ]; then
        evo_ape tum $dataset_name/groundtruth.txt logs/tum2/no_calib/$dataset/$dataset.txt -as
    else
        evo_ape tum $dataset_name/groundtruth.txt logs/tum2/calib/$dataset/$dataset.txt -as
    fi
done

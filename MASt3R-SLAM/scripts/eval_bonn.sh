#!/bin/bash
#SBATCH --account=3dv
#SBATCH --partition=jobs
#SBATCH --time=720
#SBATCH --output=out/eval_bonn_%j.out

cd /work/courses/3dv/24/MASt3R-SLAM
source /home/tdieudonne/.bashrc
conda activate slam

echo "Starting evaluation of Bonn dataset at: $(date)"

dataset_path="datasets/bonn/"
datasets=(
    #rgbd_bonn_removing_nonobstructing_box2
    #rgbd_bonn_removing_obstructing_box
    #rgbd_bonn_moving_nonobstructing_box2
    #rgbd_bonn_crowd
    #rgbd_bonn_kidnapping_box
    #rgbd_bonn_balloon
    #rgbd_bonn_placing_nonobstructing_box
    #rgbd_bonn_synchronous
    rgbd_bonn_static_close_far
    rgbd_bonn_synchronous2
    rgbd_bonn_static
    rgbd_bonn_moving_nonobstructing_box
    rgbd_bonn_balloon_tracking
    rgbd_bonn_removing_nonobstructing_box
    rgbd_bonn_moving_obstructing_box
    rgbd_bonn_person_tracking
    rgbd_bonn_placing_nonobstructing_box2
    rgbd_bonn_crowd2
    rgbd_bonn_person_tracking2
    rgbd_bonn_crowd3
    rgbd_bonn_placing_nonobstructing_box3
    rgbd_bonn_balloon2
    rgbd_bonn_moving_obstructing_box2
    rgbd_bonn_balloon_tracking2
    rgbd_bonn_placing_obstructing_box
    rgbd_bonn_kidnapping_box2
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

if [ "$print_only" = false ]; then
    for dataset in ${datasets[@]}; do
        dataset_name="$dataset_path""$dataset"/
        if [ "$no_calib" = true ]; then
            python main.py --dataset $dataset_name --no-viz --save-as bonn/no_calib/$dataset --config config/eval_no_calib.yaml
        else
            python main.py --dataset $dataset_name --no-viz --save-as bonn/calib/$dataset --config config/eval_calib.yaml
        fi
    done
fi


for dataset in ${datasets[@]}; do
    dataset_name="$dataset_path""$dataset"/
    echo ${dataset_name}
    if [ "$no_calib" = true ]; then
        python scripts/prepare_bonn.py --groundtruth_path $dataset_name/groundtruth.txt --estimated_path logs/bonn/no_calib/$dataset/$dataset.txt --output_path logs/bonn_transformed/no_calib/$dataset/$dataset.txt
    else
        python scripts/prepare_bonn.py --groundtruth_path $dataset_name/groundtruth.txt --estimated_path logs/bonn/calib/$dataset/$dataset.txt --output_path logs/bonn_transformed/calib/$dataset/$dataset.txt
    fi
done

echo "🎉 All the estimated trajectories have been transformed and saved."

for dataset in ${datasets[@]}; do
    dataset_name="$dataset_path""$dataset"/
    echo ${dataset_name}
    if [ "$no_calib" = true ]; then
        evo_ape tum $dataset_name/groundtruth.txt logs/bonn_transformed/no_calib/$dataset/$dataset.txt -as --align --pose_relation trans_part
    else
        evo_ape tum $dataset_name/groundtruth.txt logs/bonn_transformed/calib/$dataset/$dataset.txt -as --align --pose_relation trans_part
    fi
done
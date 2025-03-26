import numpy as np
import quaternion
import os
import argparse

T_m = np.matrix([
    [1.0157, 0.1828, -0.2389, 0.0113],
    [0.0009, -0.8431, -0.6413, -0.0098],
    [-0.3009, 0.6147, -0.8085, 0.0111],
    [0, 0, 0, 1.0000]
])
T_ros = np.matrix([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

def convert_bonn(groundtruth_path, estimated_path, output_path):

    groundtruth_poses = np.loadtxt(groundtruth_path)
    estimated_poses = np.loadtxt(estimated_path)

    initial_pose_gt = np.matrix(groundtruth_poses[2])   # starts at 3rd line
    t = initial_pose_gt[0,1:4].transpose()
    q = np.quaternion(initial_pose_gt[0,7],initial_pose_gt[0,4],initial_pose_gt[0,5],initial_pose_gt[0,6])
    R = quaternion.as_rotation_matrix(q)
    T_0 = np.block([[R, t], [0, 0, 0, 1]])
    T_g = T_ros * T_0 * T_ros * T_m     # inv(T_ros) = T_ros

    transformed_poses = []
    for pose in estimated_poses:
        timestamp = pose[0]
        t = np.array([[pose[1]], [pose[2]], [pose[3]]])  # Translation
        q = np.quaternion(pose[7], pose[4], pose[5], pose[6])  # Quaternion (w, x, y, z)
        R = quaternion.as_rotation_matrix(q)  # Convert quaternion to rotation matrix

        # Pose matrix 4x4
        T_est = np.block([[R, t], [0, 0, 0, 1]])

        # Apply transformation
        T_est = T_g @ T_est

        t_new = np.asarray(T_est[:3, 3]).squeeze()
        R_new = T_est[:3, :3]
        q_new = quaternion.from_rotation_matrix(R_new)
        
        transformed_poses.append([timestamp, t_new[0], t_new[1], t_new[2], q_new.x, q_new.y, q_new.z, q_new.w])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savetxt(output_path, transformed_poses, fmt='%f')
    print(f"✅ Transformed: {estimated_path} -> {output_path}")


if __name__ == "__main__":
    # python scripts/prepare_bonn.py --groundtruth_path /work/courses/3dv/24/MASt3R-SLAM/datasets/bonn/rgbd_bonn_balloon/groundtruth.txt --estimated_path /work/courses/3dv/24/MASt3R-SLAM/logs/bonn/calib/rgbd_bonn_balloon/rgbd_bonn_balloon.txt --output_path /work/courses/3dv/24/MASt3R-SLAM/logs/bonn_transformed/calib/rgbd_bonn_balloon/rgbd_bonn_balloon.txt
    parser = argparse.ArgumentParser()
    parser.add_argument("--groundtruth_path", type=str)
    parser.add_argument("--estimated_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    # Convert a model from the reference frame of the sensor to the one of the groundtruth
    convert_bonn(args.groundtruth_path, args.estimated_path, args.output_path)

#!/usr/bin/env python3
"""
Threshold Analysis Script for MASt3R-SLAM Dynamic Mask Detection

This script analyzes the error maps saved during dynamic mask computation
to help you choose optimal thresholds for your specific dataset.

Usage:
    python analyze_thresholds.py --dataset unknown_dataset --video unknown_video
    python analyze_thresholds.py --dataset tum --video rgbd_dataset_freiburg1_room --frames 10-50
"""

import argparse
import sys
import os

# Add the current directory to Python path so we can import from mast3r_slam
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mast3r_slam.archive.dynamic_mask_utils import analyze_threshold_performance


def main():
    parser = argparse.ArgumentParser(description='Analyze dynamic mask thresholds from error maps')
    parser.add_argument('--dataset', default='unknown_dataset', 
                       help='Dataset name (default: unknown_dataset)')
    parser.add_argument('--video', default='unknown_video',
                       help='Video/sequence name (default: unknown_video)')
    parser.add_argument('--frames', default=None,
                       help='Frame range to analyze (e.g., "10-50")')
    
    args = parser.parse_args()
    
    # Parse frame range if provided
    frame_range = None
    if args.frames:
        try:
            start, end = map(int, args.frames.split('-'))
            frame_range = (start, end)
            print(f"Analyzing frames {start} to {end}")
        except ValueError:
            print(f"Invalid frame range format: {args.frames}. Use format like '10-50'")
            return 1
    
    print(f"Analyzing error maps for dataset: {args.dataset}, video: {args.video}")
    
    # Run the analysis
    results = analyze_threshold_performance(
        dataset_name=args.dataset,
        video_name=args.video,
        frame_range=frame_range
    )
    
    if results is None:
        print("Analysis failed. Make sure you have run SLAM with debug_save_error_maps enabled.")
        return 1
    
    print("\nAnalysis complete!")
    print(f"You can find detailed error map visualizations in:")
    print(f"  logs/{args.dataset}/{args.video}/debug_error_maps/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main()) 
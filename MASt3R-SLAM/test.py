#!/usr/bin/env python3

import sys
print('Python path before import:')
for path in sys.path[:5]:
    print('  ', path)
print()

from mast3r_slam.easi3r_utils import easi3r_global_aligner, easi3r_GlobalAlignerMode
print('Successfully imported Easi3R global_aligner')

# Test that the function supports use_atten_mask parameter
import inspect
sig = inspect.signature(easi3r_global_aligner)
print(f'easi3r_global_aligner signature: {sig}')

# Check if the Easi3R version supports use_atten_mask
try:
    # Import the Easi3R optimizer to check for use_atten_mask support
    from dust3r.cloud_opt.optimizer import PointCloudOptimizer
    init_sig = inspect.signature(PointCloudOptimizer.__init__)
    if 'use_atten_mask' in init_sig.parameters:
        print('✓ Easi3R PointCloudOptimizer supports use_atten_mask parameter')
    else:
        print('✗ PointCloudOptimizer does not support use_atten_mask parameter')
        print(f'Available parameters: {list(init_sig.parameters.keys())}')
except Exception as e:
    print(f'Error checking PointCloudOptimizer: {e}')
#!/bin/bash

# Define the list of delta values
# deltas=(1e-3 5e-3 1e-2 5e-2 1e-1 5e-1 1 5 10 50 100 500 1000 5000 10000 50000 1e5)
# deltas=(5e5 1e6 5e6)
# # Iterate over each delta value
# for delta in "${deltas[@]}"; do
#     # Execute the Python script with the current delta value
#     python setup.py build_ext --inplace&&python ogb_exp.py --dataset mag --layer 4 --hidden 1024 --alpha 0.1 --dropout 0.3 --alg instant --cl_alg PGL --epochs 100 --delta $delta
# done

python setup.py build_ext --inplace&&python ogb_exp.py --dataset mag --layer 4 --hidden 1024 --alpha 0.1 --dropout 0.3 --alg instant --cl_alg PGL --rbmax 1 --delta 1

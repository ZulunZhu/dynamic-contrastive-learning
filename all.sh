
python ogb_exp.py --dataset products --layer 4 --hidden 1024 --alpha 0.1 --dropout 0.3 --alg instant --cl_alg PGL --epochs 200


python ogb_exp.py --dataset arxiv --layer 4 --hidden 1024 --alpha 0.1 --dropout 0.3 --alg instant --cl_alg PGL --use_gcl no --epochs 100

python ogb_exp.py --dataset mag --layer 4 --hidden 1024 --alpha 0.1 --dropout 0.3 --alg instant --cl_alg PGL --use_gcl no --epochs 100

python ogb_exp.py --dataset products --layer 4 --hidden 1024 --alpha 0.1 --dropout 0.3 --alg instant --cl_alg PGL --use_gcl no --epochs 100
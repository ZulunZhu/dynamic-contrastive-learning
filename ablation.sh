# for rbmax in 0.5 1 2 5 10; do
# for delta in 0.5 1 2 5 10; do
# python ogb_exp.py --dataset arxiv --layer 4 --hidden 1024 --alpha 0.1 --dropout 0.3 --alg instant --cl_alg PGL --rbmax $rbmax --delta $delta
# done
# done
# rbmax=( 0.5 1 2 5 10 )
# delta=( 0.5 1 2)

# for rb in "${rbmax[@]}"
# do 
#     for de in "${delta[@]}"
#     do 
#         python ogb_exp.py --dataset arxiv --layer 4 --hidden 1024 --alpha 0.1 --dropout 0.3 --alg instant --cl_alg PGL --rbmax ${rb} --delta ${de}
#     done
# done    

hiddens=(128 256 512 1024 2048 )
epochs=(1 5 20 50 100)

for hidden in "${hiddens[@]}"
do 
    for epoch in "${epochs[@]}"
    do 
        python ogb_exp.py --dataset arxiv --layer 4 --hidden ${hidden} --n-ggd-epochs ${epoch} --alpha 0.1 --dropout 0.3 --alg instant --cl_alg PGL --rbmax 1 --delta 1
    done
done
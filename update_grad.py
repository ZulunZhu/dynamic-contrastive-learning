import torch

def grad_to_vector(loaded_grad):
    # params = {n: p for n, p in loaded_grad.items()}
    vec = []
    for n,p in loaded_grad.items():
        if p is not None:
            vec.append(p.view(-1))
        else:
            # Part of the network might has no grad, fill zero for those terms
            vec.append(p.data.clone().fill_(0).view(-1))
    return torch.cat(vec)
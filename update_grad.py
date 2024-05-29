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

def get_grad(current_grad, previous_avg_grad):
        #print(memories_np.shape, gradient_np.shape)
        dotp = (current_grad * previous_avg_grad).sum()
        ref_mag = (previous_avg_grad * previous_avg_grad).sum()
        new_grad = current_grad - ((dotp / ref_mag) *  previous_avg_grad)
        # print("dotp / ref_mag", dotp / ref_mag)
        return new_grad.cuda()
def vector_to_grad(classifier, vec):
        # Overwrite current param.grad by slicing the values in vec (flatten grad)
        params = {n: p for n, p in classifier.named_parameters() if p.requires_grad}
        pointer = 0
        for n, p in params.items():
            # The length of the parameter
            num_param = p.numel()
            if p.grad is not None:
                # print("p.grad1",p.grad)
                # print("vec[pointer:pointer + num_param].view_as(p)", vec[pointer:pointer + num_param].view_as(p))
                # Slice the vector, reshape it, and replace the old data of the grad
                p.grad.copy_(vec[pointer:pointer + num_param].view_as(p))
                # Part of the network might has no grad, ignore those terms
                # print("p.grad2",p.grad)
            # Increment the pointer
            pointer += num_param
            # print("pointer", pointer)
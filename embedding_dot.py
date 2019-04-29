import math
from torch import nn
from torch.autograd import Function
import torch

if torch.cuda.is_available():
    import embedding_dot_cuda

class EmbeddingDotFunction(Function):
    @staticmethod
    def forward(ctx, embedding1, embedding2, indices):
        ctx.save_for_backward(embedding1, embedding2, indices)
        if indices.shape[0] == 0:
            # There are no indices. This needs some special casing
            return torch.zeros(size=(0,), dtype=embedding1.dtype, device=embedding1.device)
        return embedding_dot_cuda.forward(embedding1, embedding2, indices)

    @staticmethod
    def backward(ctx, grad_output):
        embedding1, embedding2, indices = ctx.saved_variables
        if indices.shape[0] == 0:
            return torch.zeros_like(embedding1), torch.zeros_like(embedding2), None
        embedding1_grad, embedding2_grad = embedding_dot_cuda.backward(grad_output, embedding1, embedding2, indices)
        return embedding1_grad, embedding2_grad, None

def embedding_dot(embedding1, embedding2, indices):
    if embedding1.is_cuda:
        return EmbeddingDotFunction.apply(embedding1, embedding2, indices)
    else:
        A = nn.functional.embedding(indices[:, 0], embedding1)
        B = nn.functional.embedding(indices[:, 1], embedding2)

        return torch.sum(A * B, dim=1)

import argparse
import numpy as np
import torch
import torch.nn as nn
import math
import time

import embedding_dot

torch.manual_seed(42)

runs = 100

time_scale = 'us'
TIME_SCALES = {'s': 1, 'ms': 1000, 'us': 1000000}

embedding1 = torch.randn(30000, 700, requires_grad=True, device='cuda:0')
embedding2 = torch.randn(30000, 700, requires_grad=True, device='cuda:0')

indices = torch.randint(high=30, size=(30000, 2), device='cuda:0')

def baseline_embedding_dot(embedding1, embedding2, indices):
    A = nn.functional.embedding(indices[:, 0], embedding1)
    B = nn.functional.embedding(indices[:, 1], embedding2)

    return torch.sum(A * B, dim=1)

def cuda_embedding_dot(embedding1, embedding2, indices):
    return embedding_dot.embedding_dot(embedding1, embedding2, indices)

# Force CUDA initialization
output = baseline_embedding_dot(embedding1, embedding2, indices)
output.sum().backward()

def test(func):
    forward_min = math.inf
    forward_time = 0
    backward_min = math.inf
    backward_time = 0

    for _ in range(100):
        embedding1.grad.zero_()
        embedding2.grad.zero_()
        output = func(embedding1, embedding2, indices)
        output.sum().backward()

    for _ in range(runs):
        embedding1.grad.zero_()
        embedding2.grad.zero_()


        torch.cuda.synchronize()
        start = time.time()
        output = func(embedding1, embedding2, indices)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        forward_min = min(forward_min, elapsed)
        forward_time += elapsed


        torch.cuda.synchronize()
        start = time.time()
        output.sum().backward()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        backward_min = min(backward_min, elapsed)
        backward_time += elapsed

    scale = TIME_SCALES[time_scale]
    forward_min *= scale
    backward_min *= scale
    forward_average = forward_time / runs * scale
    backward_average = backward_time / runs * scale

    print('Forward: {0:.3f}/{1:.3f} {4} | Backward {2:.3f}/{3:.3f} {4}'.format(
        forward_min, forward_average, backward_min, backward_average,
        time_scale))

test(baseline_embedding_dot)
test(cuda_embedding_dot)
test(baseline_embedding_dot)
test(cuda_embedding_dot)


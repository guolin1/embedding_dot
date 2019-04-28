import argparse
import numpy as np
import torch
import torch.nn as nn

import embedding_dot

torch.manual_seed(42)

embedding1 = torch.randn(3000, 700, requires_grad=True, device='cuda:0')
embedding2 = torch.randn(3000, 700, requires_grad=True, device='cuda:0')

indices = torch.randint(high=200, size=(0, 2), device='cuda:0')

def baseline_embedding_dot(embedding1, embedding2, indices):
	A = nn.functional.embedding(indices[:, 0], embedding1)
	B = nn.functional.embedding(indices[:, 1], embedding2)

	return torch.sum(A * B, dim=1)

def cuda_embedding_dot(embedding1, embedding2, indices):
	return embedding_dot.embedding_dot(embedding1, embedding2, indices)

output = baseline_embedding_dot(embedding1, embedding2, indices)
output.sum().backward()

embedding1.grad.zero_()
embedding2.grad.zero_()

baseline = baseline_embedding_dot(embedding1, embedding2, indices)
baseline.sum().backward()

# print(baseline)
print(baseline.sum())
print(embedding1.grad.sum())
print(embedding2.grad.sum())

embedding1.grad.zero_()
embedding2.grad.zero_()

other = cuda_embedding_dot(embedding1, embedding2, indices)
other.sum().backward()

# print(other)
print(other.sum())
print(embedding1.grad.sum())
print(embedding2.grad.sum())
#include <torch/extension.h>

#include <vector>
#include <iostream>

#include <c10/cuda/CUDAException.h>

#include <stdio.h>

#define FULL_MASK 0xffffffff

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
__global__ void EmbeddingDot_updateOutputKernel(
    const scalar_t* __restrict__ embedding1, const scalar_t*  __restrict__ embedding2, const int64_t * __restrict__ indices, scalar_t* __restrict__ output,
    int64_t embedding1_stride, int64_t embedding2_stride, int64_t indices_stride,
    int64_t num_features, int64_t num_indices) {

    scalar_t accum = 0;

    int64_t index = threadIdx.y + blockIdx.x * blockDim.y;

    if (index < num_indices) {
        int64_t embedding1_index = indices[indices_stride * index + 0] * embedding1_stride;
        int64_t embedding2_index = indices[indices_stride * index + 1] * embedding2_stride;

        for (int64_t featureDim = threadIdx.x; featureDim < num_features; featureDim += blockDim.x) {
            accum += embedding1[embedding1_index + featureDim] * embedding2[embedding2_index + featureDim];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            accum += __shfl_down_sync(FULL_MASK, accum, offset);
        }

        if (threadIdx.x == 0) {
            // This is the warp leader
            output[index] = accum;
        }
    }
}

torch::Tensor embedding_dot_forward(
        torch::Tensor embedding1,
        torch::Tensor embedding2,
        torch::Tensor indices) {
    CHECK_INPUT(embedding1);
    CHECK_INPUT(embedding2);
    CHECK_INPUT(indices);

    torch::Tensor output = torch::empty({indices.size(0)}, embedding1.options());

    dim3 block = dim3(32, 8);
    int grid = (indices.size(0) + 7) / 8;

    EmbeddingDot_updateOutputKernel<float><<<grid, block>>>(
        embedding1.data<float>(), embedding2.data<float>(),
        indices.data<int64_t>(), output.data<float>(),
        embedding1.stride(0), embedding2.stride(0), indices.stride(0),
        embedding1.size(1), indices.size(0));
    

    C10_CUDA_CHECK(cudaGetLastError());

    return output;
}

template <typename scalar_t>
__global__ void EmbeddingDot_updateGradKernel(
    const scalar_t* __restrict__ embedding1, const scalar_t* __restrict__ embedding2, scalar_t* embedding1_grad, scalar_t* __restrict__ embedding2_grad, 
    const int64_t* __restrict__ indices, const scalar_t* __restrict__ output_grad,
    int64_t embedding_stride, int64_t indices_stride,
    int64_t num_features, int64_t num_indices) {

    int64_t index = threadIdx.y + blockIdx.x * blockDim.y;
    int64_t side = blockIdx.y;

    if (index < num_indices) {
        scalar_t index_grad = output_grad[index];

        int64_t source_index = indices[indices_stride * index + side] * embedding_stride;
        int64_t destination_index = indices[indices_stride * index + 1 - side] * embedding_stride;

        const scalar_t* source_location = (side ? embedding2 : embedding1) + source_index;
        scalar_t* target_location = (side ? embedding1_grad : embedding2_grad) + destination_index;

        for (int64_t featureDim = threadIdx.x; featureDim < num_features; featureDim += blockDim.x) {
            scalar_t value = source_location[featureDim] * index_grad;
            atomicAdd(target_location + featureDim, value);
        }
    }
}

std::vector<torch::Tensor> embedding_dot_backward(
        torch::Tensor grad_output_,
        torch::Tensor embedding1,
        torch::Tensor embedding2,
        torch::Tensor indices) {

    torch::Tensor grad_output = grad_output_.contiguous();

    CHECK_INPUT(grad_output);
    CHECK_INPUT(embedding1);
    CHECK_INPUT(embedding2);
    CHECK_INPUT(indices);

    torch::Tensor embedding1_grad = torch::zeros_like(embedding1);
    torch::Tensor embedding2_grad = torch::zeros_like(embedding2);

    dim3 block = dim3(32, 8);
    dim3 grid = dim3((indices.size(0) + 7) / 8, 2);

    EmbeddingDot_updateGradKernel<float><<<grid, block>>>(
        embedding1.data<float>(), embedding2.data<float>(),
        embedding1_grad.data<float>(), embedding2_grad.data<float>(),
        indices.data<int64_t>(), grad_output.data<float>(),
        embedding1.stride(0), indices.stride(0),
        embedding1_grad.size(1), indices.size(0));

    C10_CUDA_CHECK(cudaGetLastError());

    std::vector<torch::Tensor> result;
    result.emplace_back(std::move(embedding1_grad));
    result.emplace_back(std::move(embedding2_grad));

    return result;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &embedding_dot_forward, "Embedding dot forward (CUDA)");
  m.def("backward", &embedding_dot_backward, "Embedding dot backward (CUDA)");
}

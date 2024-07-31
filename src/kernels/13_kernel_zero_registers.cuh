#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

// CUDA kernel written by Arkadiusz Paterek
// FP32 matrix multiplication

const int BK = 16;

__global__ void sgemmZeroRegisters(int M, int N, int K, float alpha, const float *A,
                                   const float *B, float beta, float *C) {
	int row = 16*(blockIdx.y * blockDim.y + threadIdx.y);
	int col = 8*(blockIdx.x * blockDim.x + threadIdx.x);
	float sum[64] = {0};
	__shared__ float aTile[16][16][BK];
	__shared__ float bTile[8][BK][32];

	for (int k0 = 0; k0 < K; k0 += BK) {
		#pragma unroll
		for (int y2 = 0; y2 < 16; ++y2)
			aTile[y2][threadIdx.y][threadIdx.x >> 1] = A[(row + y2)*K + k0 + (threadIdx.x >> 1)];

		#pragma unroll
		for (int x2 = 0; x2 < 8; ++x2)
			bTile[x2][threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y)*N + col + x2];
		__syncthreads();

		#pragma unroll
		for (int y2 = 0; y2 < 16; ++y2)
			for (int x2 = 0; x2 < 8; ++x2)
				for (int k = 0; k < BK; ++k)
					sum[y2*8 + x2] += aTile[y2][threadIdx.y][k] * bTile[x2][k][threadIdx.x];
		__syncthreads();
	}

	for (int y2 = 0; y2 < 16; ++y2)
		for (int x2 = 0; x2 < 8; ++x2)
			C[(row + y2)*N + col + x2] = alpha * sum[y2*8 + x2] + beta * C[(row + y2)*N + col + x2];
}
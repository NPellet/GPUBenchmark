#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include <cassert>

template<typename T>
__device__ inline T getMatrixIndex(const T& x, const T& y, const T& w) { // Will likely have to go through type conversion upon calling
	return w * y + x;
}

__host__ __device__ inline uint32_t pow2(const int32_t& a) {
	return a * a;
}

template<typename T>
__host__ __device__ inline float dot(const T& rep, const float repMag, const int32_t x, const int32_t y) {
	return (float)(rep.x * x + rep.y * y) / repMag;
}

__host__ __device__ inline float mag(const int2& rep) {
	return sqrt((float)mag(rep));
}
__host__ __device__ inline float mag2(const int2& rep) {
	return pow2(rep.x) + pow2(rep.y);
}

template<typename TRET, typename T>
TRET __host__ __device__ findVectorComponents(int x, int y, T vec_rep_1, T vec_rep_2) {
	TRET ret;
	float m = ((float)(vec_rep_1.y * x - vec_rep_1.x * y) / (float)(vec_rep_2.x * vec_rep_1.y - vec_rep_2.y * vec_rep_1.x));
	ret.y = static_cast<decltype(ret.x)>(m);

	if (vec_rep_1.x == 0) {
		ret.x = decltype(ret.x)((float)(y - m * vec_rep_2.y) / (vec_rep_1.y));
	}
	else {
		ret.x = decltype(ret.x)((float)(x - m * vec_rep_2.x) / (vec_rep_1.x));
	}

	return ret;
}

inline cudaError_t checkCuda(cudaError_t result) {
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

inline cudaError_t checkCudaKernel(const char* error = nullptr) {
#if defined(DEBUG) || defined(_DEBUG)
	cudaError_t cudaError;
	if ((cudaError = cudaGetLastError()) != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s %s\n", error, cudaGetErrorString(cudaError));
		assert(cudaError == cudaSuccess);
	}
	return cudaError;
#endif
	return cudaSuccess;
}
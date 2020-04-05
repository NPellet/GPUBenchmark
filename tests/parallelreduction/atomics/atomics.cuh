#pragma once
/**
*	Author: Norman Pellet
*	Date: April 2nd, 2020
*
*/
#include <stdint.h>

#include "../test_parallelred.hpp"
#include "../../../util/gpuutil.cuh"

namespace Tests {
	namespace ParallelReduction {
		template<typename T>
		__global__ void kernelSumAll(
			const T* source,
			const uint32_t w,
			const uint32_t h,
			T* sum
			) {
			const int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
			const int32_t row = blockIdx.y * blockDim.y + threadIdx.y;

			if (col >= w || row >= h) {
				return;
			}

			const size_t index = getMatrixIndex<size_t>(col, row, w);
			atomicAdd((T*)sum, (T)source[index]);
		}

		template<typename T>
		__global__ void kernelSumAll_Seq(
			const T* source,
			const uint32_t l,
			T* sum
			) {
			const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;

			if (index >= l) {
				return;
			}

			atomicAdd((T*)sum, (T)source[index]);
		}

		template<typename T>
		__global__ void kernelSumAll_shared(
			const T* source,
			const uint32_t w,
			const uint32_t h,
			T* sum
			) {
			__shared__ T sSum;
			if (threadIdx.x == 0 && threadIdx.y == 0) {
				sSum = 0;
			}
			__syncthreads();

			const int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
			const int32_t row = blockIdx.y * blockDim.y + threadIdx.y;

			if (col >= w || row >= h) {
				return;
			}

			const size_t index = getMatrixIndex<size_t>(col, row, w);
			atomicAdd_block(&sSum, source[index]);

			__syncthreads();

			if (threadIdx.x == 0 && threadIdx.y == 0) {
				atomicAdd(sum, sSum);
			}
		}

		class GPUAtomics : public Base {
		private:
		public:

			GPUAtomics(const std::string source, const std::string descr, const uint32_t num) : Base(source, descr, num) {}

		protected:

			void testSetup() override {
				Base::testSetup();
			};

			void testExecute() {
				zType sum = 0;
				zType* deviceSum;
				checkCuda(cudaMalloc(&deviceSum, sizeof(zType)));
				checkCuda(cudaMemcpy(deviceSum, &sum, sizeof(zType), cudaMemcpyHostToDevice));

				kernelSumAll< zType> << < source_rescaled->dimGrid, source_rescaled->dimBlock >> > (
					source_rescaled->gpuGetPtr(),
					source_rescaled->getWidth(),
					source_rescaled->getHeight(),

					deviceSum
					);
				checkCudaKernel();
				cudaDeviceSynchronize();

				checkCuda(cudaMemcpy(&sum, deviceSum, sizeof(zType), cudaMemcpyDeviceToHost));
				verificationValue = static_cast<uint32>(sum);
			}

		public:
			void testExecute_smem() {
				zType sum = 0;
				zType* deviceSum;
				checkCuda(cudaMalloc(&deviceSum, sizeof(zType)));
				checkCuda(cudaMemcpy(deviceSum, &sum, sizeof(zType), cudaMemcpyHostToDevice));

				kernelSumAll_shared< zType> << < source_rescaled->dimGrid, source_rescaled->dimBlock >> > (
					source_rescaled->gpuGetPtr(),
					source_rescaled->getWidth(),
					source_rescaled->getHeight(),

					deviceSum
					);
				checkCudaKernel();
				cudaDeviceSynchronize();

				checkCuda(cudaMemcpy(&sum, deviceSum, sizeof(zType), cudaMemcpyDeviceToHost));
				verificationValue = static_cast<uint32>(sum);
			}

			void testExecute_seq() {
				zType sum = 0;
				zType* deviceSum;
				checkCuda(cudaMalloc(&deviceSum, sizeof(zType)));
				checkCuda(cudaMemcpy(deviceSum, &sum, sizeof(zType), cudaMemcpyHostToDevice));

				kernelSumAll_Seq< zType> << <  (source_rescaled->getWidth() * source_rescaled->getHeight() + 1023) / 1024, 1024 >> > (
					source_rescaled->gpuGetPtr(),
					source_rescaled->getWidth() * source_rescaled->getHeight(),

					deviceSum
					);
				checkCudaKernel();
				cudaDeviceSynchronize();

				checkCuda(cudaMemcpy(&sum, deviceSum, sizeof(zType), cudaMemcpyDeviceToHost));
				verificationValue = static_cast<uint32>(sum);
			}
		};
	}
}
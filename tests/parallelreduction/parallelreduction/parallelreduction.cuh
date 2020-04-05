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
		__global__ void kernelParallelReduce(
			const T* input,
			const uint64 width,
			const uint64 height,
			T* reduced
			) {
			// Start loading shared memory with content of the thread block
			extern __shared__ T sMem[];

			// Reduced for the block data is offseted in the same shared memory blockLength

			const uint32_t blockLength = blockDim.x * blockDim.y;
			// Sequential IDs are important. also, x is major, because of how a warp is arranged
			const uint32_t threadId = threadIdx.x + blockDim.x * threadIdx.y;

			// Get matrix coordinates
			const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
			const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;

			if (col < width && row < height) { // Copy global memory into shared memory
				sMem[threadId] = input[getMatrixIndex<size_t>(col, row, width)];
			}
			else {
				sMem[threadId] = 0;
			}

			__syncthreads();
			// End loading shared memory with content of the thread block
			 // Shared memory is ready to be reduced
			// We now have a 1D array of size blockLength which contains the global data

			for (unsigned int s = blockLength / 2; s >= 1; s >>= 1) {
				if (threadId < s) {
					sMem[threadId] = sMem[threadId] + sMem[threadId + s]; // Reduction step
				}
				__syncthreads();
			}

			// For the one thread into which the data has been reduced
			if (threadIdx.x == 0 && threadIdx.y == 0) {
				// Note how we are filling a global matrix of size (gridDim.x, gridDim.y) at the position (blockId.x, blockId.y)
				reduced[blockIdx.y * gridDim.x + blockIdx.x] = sMem[0];
			}
		}

		template<typename T, uint32_t unrollNum>
		__device__ inline __forceinline__  void loadIntoSMem(
			volatile T* sMem,
			const T* input,
			const uint32_t col,
			const uint32_t row,
			const uint32_t strideY,
			const uint32_t width,
			const uint32_t height
			) {
			if (row < height) {
				*sMem += input[getMatrixIndex<size_t>(col, row, width)];
			}

			//if (!std::equal_to<uint32_t>(unrollNum, 2)) {
			if constexpr (unrollNum > 2) {
				loadIntoSMem<T, unrollNum - 1>(sMem, input, col, row + strideY, strideY, width, height);
			}
		}

		template<typename T, uint32_t numLoads = 1>
		__global__ void kernelParallelReduce_multLoad(
			const T* input,
			const uint64 width,
			const uint64 height,
			T* reduced
			) {
			// Start loading shared memory with content of the thread block
			extern __shared__ T sMem[];

			// Reduced for the block data is offseted in the same shared memory blockLength

			const uint32_t blockLength = blockDim.x * blockDim.y;
			// Sequential IDs are important. also, x is major, because of how a warp is arranged
			const uint32_t threadId = threadIdx.x + blockDim.x * threadIdx.y;

			// Get matrix coordinates
			const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
			const uint32_t row = blockIdx.y * blockDim.y * numLoads + threadIdx.y;

			if (col < width && row < height) { // Copy global memory into shared memory
				sMem[threadId] = input[getMatrixIndex<size_t>(col, row, width)];
				if constexpr (numLoads > 1) {
					loadIntoSMem<T, numLoads>(
						&(sMem[threadId]),
						input,
						col,
						row + blockDim.y,
						blockDim.y,
						width,
						height
						);
				}
			}
			else {
				sMem[threadId] = 0;
			}

			__syncthreads();
			// End loading shared memory with content of the thread block
			 // Shared memory is ready to be reduced
			// We now have a 1D array of size blockLength which contains the global data

			for (unsigned int s = blockLength / 2; s >= 1; s >>= 1) {
				if (threadId < s) {
					sMem[threadId] = sMem[threadId] + sMem[threadId + s]; // Reduction step
				}
				__syncthreads();
			}

			// For the one thread into which the data has been reduced
			if (threadIdx.x == 0 && threadIdx.y == 0) {
				// Note how we are filling a global matrix of size (gridDim.x, gridDim.y) at the position (blockId.x, blockId.y)
				reduced[blockIdx.y * gridDim.x + blockIdx.x] = sMem[0];
			}
		}

		template<typename T, uint32_t numLoads = 1>
		__global__ void kernelParallelReduce_multLoad_unroll(
			const T* input,
			const uint64 width,
			const uint64 height,
			T* reduced
			) {
			// Start loading shared memory with content of the thread block
			extern __shared__ T sMem[];

			// Reduced for the block data is offseted in the same shared memory blockLength

			const uint32_t blockLength = blockDim.x * blockDim.y;
			// Sequential IDs are important. also, x is major, because of how a warp is arranged
			const uint32_t threadId = threadIdx.x + blockDim.x * threadIdx.y;

			// Get matrix coordinates
			const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
			size_t row = blockIdx.y * blockDim.y * numLoads + threadIdx.y;

			int i = 1;
			if (col < width && row < height) { // Copy global memory into shared memory
				sMem[threadId] = input[getMatrixIndex<size_t>(col, row, width)];
				if constexpr (numLoads > 1) {
					loadIntoSMem<T, numLoads>(
						&(sMem[threadId]),
						input,
						col,
						row + blockDim.y,
						blockDim.y,
						width,
						height
						);
				}
			}
			else {
				sMem[threadId] = 0;
			}

			__syncthreads();
			// End loading shared memory with content of the thread block
			 // Shared memory is ready to be reduced
			// We now have a 1D array of size blockLength which contains the global data

			for (unsigned int s = blockLength / 2; s > 32; s >>= 1) {
				if (threadId < s) {
					sMem[threadId] = sMem[threadId] + sMem[threadId + s]; // Reduction step
				}
				__syncthreads();
			}

			// Applies to threads 0 to 31, i.e. the ones in the last warp.
			if (threadId < 32) { // First warp only, but still needs to be synchronized
				if (blockLength >= 64) {
					sMem[threadId] += sMem[threadId + 32]; __syncwarp();
				}
				if (blockLength >= 32) {
					sMem[threadId] += sMem[threadId + 16]; __syncwarp();
				}
				if (blockLength >= 16) {
					sMem[threadId] += sMem[threadId + 8]; __syncwarp();
				}
				if (blockLength >= 8) {
					sMem[threadId] += sMem[threadId + 4]; __syncwarp();
				}
				if (blockLength >= 4) {
					sMem[threadId] += sMem[threadId + 2]; __syncwarp();
				}
				if (blockLength >= 2) {
					sMem[threadId] += sMem[threadId + 1];
				}

				// For the one thread into which the data has been reduced
				if (threadId == 0) {
					// Note how we are filling a global matrix of size (gridDim.x, gridDim.y) at the position (blockId.x, blockId.y)
					reduced[blockIdx.y * gridDim.x + blockIdx.x] = sMem[0];
				}
			}
		}

		class GPUParallelReduction : public Base {
		private:
		public:

			GPUParallelReduction(const std::string source, const std::string descr, const uint32_t num) : Base(source, descr, num) {}

			template<uint32_t numLoad>
			void testExecute_multLoad() {
				parallelReduce<numLoad>([](TileZScaled* input, TileZScaled* reduced) {
					dim3 gridSize = input->dimGrid;
					dim3 blockSize = input->dimBlock;

					kernelParallelReduce_multLoad<zType, numLoad> << < gridSize, blockSize, sizeof(zType)* blockSize.x* blockSize.y >> > (
						input->gpuGetPtr(),
						input->getWidth(),
						input->getHeight(),
						reduced->gpuGetPtr()
						);
					});
			}

			template<uint32_t numLoad>
			void testExecute_multLoad_unroll() {
				parallelReduce<numLoad>([](TileZScaled* input, TileZScaled* reduced) {
					dim3 gridSize = input->dimGrid;
					dim3 blockSize = input->dimBlock;

					kernelParallelReduce_multLoad_unroll<zType, numLoad> << < gridSize, blockSize, sizeof(zType)* blockSize.x* blockSize.y >> > (
						input->gpuGetPtr(),
						input->getWidth(),
						input->getHeight(),
						reduced->gpuGetPtr()
						);
					});
			}

		protected:

			void testSetup() override {
				Base::testSetup();
			};

			void testExecute() {
				parallelReduce<1>([](TileZScaled* input, TileZScaled* reduced) {
					dim3 gridSize = input->dimGrid;
					dim3 blockSize = input->dimBlock;

					kernelParallelReduce<zType> << < { reduced->getWidth(), reduced->getHeight() }, blockSize, sizeof(zType)* blockSize.x* blockSize.y >> > (
						input->gpuGetPtr(),
						input->getWidth(),
						input->getHeight(),
						reduced->gpuGetPtr()
						);
					});
			}

			template<int numLoad = 1>
			void inline parallelReduce(std::function<void(TileZScaled*, TileZScaled*)> kernelCall) {
				TileZScaled* input = source_rescaled;
				TileZScaled* reduced;

				int i = 0;

				while (true) {
					dim3 gridSize = input->dimGrid;
					uint32_t rem = (gridSize.y % numLoad);
					if (rem > 0) {
						gridSize.y += numLoad - rem;
					}
					gridSize.y /= numLoad;

					reduced = new TileZScaled{ gridSize.x, gridSize.y };
					reduced->gpuAllocate();

					kernelCall(input, reduced);
					//	checkCudaKernel();
					cudaDeviceSynchronize();
					verificationValue = reduced->getPointXY(0, 0);

					if (i > 0) {
						if (gridSize.x == 1 && gridSize.y == 1) {
							delete input;
							break;
						}

						delete input; // for i > 0, input is equal to the reduced of the previous iteration, which can (and must) be discarded.
					}

					// Re-assign the reduced matrix as the input for the next loop
					input = reduced;

					// Used to determine which iteration we're at
					i++;
				}
				reduced->gpuRetrieve();
				verificationValue = reduced->getPointXY(0, 0);

				delete reduced; // No need for the reduced matrix
			}
		};
	}
}
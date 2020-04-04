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
			extern __shared__ T s[];

			// Reduced for the block data is offseted in the same shared memory blockLength

			const uint32_t blockLength = blockDim.x * blockDim.y;
			// Sequential IDs are important. also, x is major, because of how a warp is arranged
			const uint32_t threadId = threadIdx.x + blockDim.x * threadIdx.y;

			// Get matrix coordinates
			const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
			const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;

			if (col < matrixWidth && row < matrixHeight) { // Copy global memory into shared memory
				s[threadId] = sourceMin[getMatrixIndex<size_t>(col, row, width];
			}
			else {
				s[threadId] = 0;
			}

			__syncthreads();
			// End loading shared memory with content of the thread block
			 // Shared memory is ready to be reduced
			// We now have a 1D array of size blockLength which contains the global data

			for (unsigned int s = blockLength / 2; s > 1; s >>= 1) {
				if (threadId < s) {
					s[threadId] = s[threadId] + s[threadId + s]; // Reduction step
				}
				__syncthreads();
			}

			// For the one thread into which the data has been reduced
			if (threadIdx.x == 0 && threadIdx.y == 0) {
				// Note how we are filling a global matrix of size (gridDim.x, gridDim.y) at the position (blockId.x, blockId.y)
				reduced[blockIdx.y * gridDim.x + blockIdx.x] = s[0];
			}
		}

		class GPUParallelReduction : public Base {
		private:
		public:

			GPUParallelReduction(const std::string source, const std::string descr, const uint32_t num) : Base(source, descr, num) {}

		protected:

			void testSetup() override {
				Base::testSetup();
			};

			void testExecute() {
				parallelReduce([](TileZScaled* input, TileZScaled* reduced) {
					kernelParallelReduce<T> << < gridSize, blockSize, sizeof(T)* blockSize.x* blockSize.y >> > (
						input->gpuGetPtr(),
						input->getWidth(),
						input->getHeight(),
						reduced->gpuGetPtr()
						);
					});
			}

			void parallelReduce(std::function<void(TileZScaled*, TileZScaled*)> kernelCall) {
				source_rescaled->gpuAllocate();

				TileZScaled* input = source_rescaled;
				TileZScaled* reduced;

				int i = 0;

				while (true) {
					dim3 gridSize = input->dimGrid;
					dim3 blockSize = input->dimBlock;

					reduced = new TileZScaled{ gridSize.x, gridSize.y };
					reduced->gpuAllocate();

					kernelCall(input, reduced);

					checkCudaKernel();
					cudaDeviceSynchronize();

					if (i > 0) {
						delete input; // for i > 0, input is equal to the reduced of the previous iteraiton, which can (and must) be discarded.
						break;
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
		}
	}
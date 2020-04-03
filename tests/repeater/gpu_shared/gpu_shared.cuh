#pragma once
/**
*	Author: Norman Pellet
*	Date: April 2nd, 2020
*	Simple repeater kernel executed on the GPU. Loops over the repeater tile
*/
#include <stdint.h>
#include <string>

#include "../test_repeater.hpp"
#include "../../../util/gpuutil.cuh"

namespace Tests {
	namespace Repeater {
		template<typename T, uint32_t SIZE_REPEATER>
		__global__ void kernelRepeaterModuloShared(
			T* target,
			const uint32_t width,
			const uint32_t height,

			const T* repeater,
			const uint32_t repeaterWidth,
			const uint32_t repeaterHeight
			) {
			__shared__ T s[SIZE_REPEATER * SIZE_REPEATER];

			if (threadIdx.x == 0 && threadIdx.y == 0) {
				memcpy(s, repeater, sizeof(T) * SIZE_REPEATER * SIZE_REPEATER);
			}
			__syncthreads();

			const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
			const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;

			if (col >= width || row >= height) {
				return;
			}

			const size_t targetIndex = getMatrixIndex(col, row, width);
			const uint32_t colTarget = col % repeaterWidth;
			const uint32_t rowTarget = row % repeaterHeight;

			target[targetIndex] = s[getMatrixIndex(colTarget, rowTarget, repeaterWidth)];
		}

		template<uint32_t SIZE_REPEATER, uint32_t SIZE_GLOBAL>
		class GPURepeaterShared : public Base<SIZE_REPEATER, SIZE_GLOBAL> {
			using BaseC = Base<SIZE_REPEATER, SIZE_GLOBAL>;

		public:

			GPURepeaterShared(const std::string descr, const uint32_t num) : BaseC(descr, num) {}

		protected:

			void testSetup() override {
				BaseC::testSetup();
				repeater.gpuAllocate();
				target.gpuAllocate();
			};

			void testExecute() {
				//repeater.gpuAllocate(true, true);
				//target.gpuAllocate(true, true);

				kernelRepeaterModuloShared<float, SIZE_REPEATER> << < target.dimGrid, target.dimBlock >> > (
					target.gpuGetPtr(),
					target.getWidth(),
					target.getHeight(),

					repeater.gpuGetPtr(),
					repeater.getWidth(),
					repeater.getHeight()
					);

				cudaDeviceSynchronize();

				//target.gpuRetrieve();
			}
		};
	}
}
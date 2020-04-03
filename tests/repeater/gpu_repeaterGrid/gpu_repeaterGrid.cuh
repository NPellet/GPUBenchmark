#pragma once
/**
*	Author: Norman Pellet
*	Date: April 2nd, 2020
*	Simple repeater kernel executed on the GPU. Loops over the repeater tile
*/
#include <stdint.h>
#include "../test_repeater.hpp"
#include "../../../util/gpuutil.cuh"

namespace Tests {
	namespace Repeater {
		template<typename T>
		__global__ void kernelRepeaterRepGrid(
			T* target,
			const uint32_t width,
			const uint32_t height,

			const T* repeater,
			const uint32_t repeaterWidth,
			const uint32_t repeaterHeight
			) {
			const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
			const uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;

			if (col >= repeaterWidth || row >= repeaterHeight) {
				return;
			}

			const T repeaterValue = repeater[getMatrixIndex(col, row, repeaterWidth)];

			for (int y = row; y < height; y += repeaterHeight) {
				size_t index = getMatrixIndex<size_t>(col, y, width);
				for (int x = col; x < width; x += repeaterWidth) {
					target[index] = repeaterValue;
					index += repeaterWidth;
				}
			}
		}

		template<uint32_t SIZE_REPEATER, uint32_t SIZE_GLOBAL>
		class GPURepeaterRepGrid : public Base<SIZE_REPEATER, SIZE_GLOBAL> {
			using BaseC = Base<SIZE_REPEATER, SIZE_GLOBAL>;

		public:

			GPURepeaterRepGrid(const std::string descr, const uint32_t num) : BaseC(descr, num) {}

		protected:

			void testSetup() override {
				BaseC::testSetup();
				repeater.gpuAllocate();
				target.gpuAllocate();
			};

			void testExecute() {
				kernelRepeaterRepGrid << < repeater.dimGrid, repeater.dimBlock >> > (
					target.gpuGetPtr(),
					target.getWidth(),
					target.getHeight(),

					repeater.gpuGetPtr(),
					repeater.getWidth(),
					repeater.getHeight()
					);

				cudaDeviceSynchronize();
			}
		};
	}
}
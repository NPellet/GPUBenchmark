#pragma once
/**
*	Author: Norman Pellet
*	Date: April 2nd, 2020
*	Simple repeater kernel executed on the GPU. Loops over the target tile
*/

#include "../test_repeater.hpp"
#include <stdint.h>
#include <string>
#include "../../../util/gpuutil.cuh"

namespace Tests {
	namespace Repeater {
		template<typename T, typename U>
		__global__ void kernelRepeaterModulo(
			T* target,
			const uint32_t width,
			const uint32_t height,

			const T* repeater,
			const uint32_t repeaterWidth,
			const uint32_t repeaterHeight
			) {
			const U col = blockIdx.x * blockDim.x + threadIdx.x;
			const U row = blockIdx.y * blockDim.y + threadIdx.y;

			if (col >= width || row >= height) {
				return;
			}

			const U targetIndex = getMatrixIndex<U>(col, row, width);
			const uint32_t colTarget = col % repeaterWidth;
			const uint32_t rowTarget = row % repeaterHeight;

			target[targetIndex] = repeater[getMatrixIndex<U>(colTarget, rowTarget, repeaterWidth)];
		}

		template<uint32_t SIZE_REPEATER, uint32_t SIZE_GLOBAL, bool includeCopy, typename TYPE_INDEX>
		class GPURepeater : public Base<SIZE_REPEATER, SIZE_GLOBAL> {
			using BaseC = Base<SIZE_REPEATER, SIZE_GLOBAL>;

		public:

			GPURepeater(const std::string descr, const uint32_t num) : BaseC(descr, num) {}

		protected:

			void testSetup() override {
				BaseC::testSetup();
				repeater.gpuAllocate();
				target.gpuAllocate();
			};

			void testExecute() {
				if (includeCopy) { // Compile time evaluation
					repeater.gpuAllocate(true, true);
					//target.gpuAllocate(true, true);
				}

				kernelRepeaterModulo<float, TYPE_INDEX> << < target.dimGrid, target.dimBlock >> > (
					target.gpuGetPtr(),
					target.getWidth(),
					target.getHeight(),

					repeater.gpuGetPtr(),
					repeater.getWidth(),
					repeater.getHeight()
					);

				cudaDeviceSynchronize();
				if (includeCopy) { // Compile time evaluation
					target.gpuRetrieve();
				}
			}
		};
	}
}
#pragma once
/**
*	Author: Norman Pellet
*	Date: April 2nd, 2020
*	UV repeater code executed on the GPU. Looping over UV coordinates
*/
#include "../test_repeater.hpp"
#include "../../../util/gpuutil.cuh"
#include <stdint.h>

namespace Tests {
	namespace RepeaterGeneral {
		template<typename T, uint32_t SIZE_GLOBAL, uint32_t SIZE_REPEATER, bool exch>
		void __global__ kernelRepeaterUVLoop(
			T* target,
			const uint32_t width,
			const uint32_t height,

			const T* repeater,
			const uint32_t repeaterWidth,
			const uint32_t repeaterHeight,

			const int2 u,
			const int2 v,

			const int32_t minU,
			const int32_t minV
			) {
			const int32_t gridU = blockIdx.x * blockDim.x + threadIdx.x + minU;
			const int32_t gridV = blockIdx.y * blockDim.y + threadIdx.y + minV;

			const int32_t startX = gridU * u.x + gridV * v.x;
			const int32_t startY = gridU * u.y + gridV * v.y;

			size_t indexRepeater = 0;

			for (uint32_t y = 0; y < SIZE_REPEATER; y++) {
				for (uint32_t x = 0; x < SIZE_REPEATER; x++) {
					float repValue = repeater[indexRepeater];
					indexRepeater++;

					if (startX + x < 0 || startY + y < 0 || startX + x >= SIZE_GLOBAL || startY + y >= SIZE_GLOBAL) {
						continue;
					}

					const size_t index = getMatrixIndex(startX + x, startY + y, SIZE_GLOBAL);
					if (repValue != 0) {
						if (exch) {
							atomicExch(target + index, repValue);// target[index] = repValue;
						}
						else {
							target[index] = repValue;
						}
					}
				}
			}
		}

		template<uint32_t SIZE_REPEATER, uint32_t SIZE_GLOBAL, bool exchange>
		class GPUUV : public Base<SIZE_REPEATER, SIZE_GLOBAL> {
		public:

			GPUUV(const std::string descr, const uint32_t num) : Base<SIZE_REPEATER, SIZE_GLOBAL>(descr, num) {}

			void testSetup() {
				Base<SIZE_REPEATER, SIZE_GLOBAL>::testSetup();
				target.fillWith(0);
				target.gpuAllocate(true, true);
				repeater.gpuAllocate();
			}

			void testExecute() {
				updateBoundaries(0, 0);
				updateBoundaries(SIZE_GLOBAL, 0);
				updateBoundaries(0, SIZE_GLOBAL);
				updateBoundaries(SIZE_GLOBAL, SIZE_GLOBAL);

				assert(uMax >= uMin);
				assert(vMax >= vMin);

				dim3 dimBlock{ 32, 32 };
				dim3 dimGrid{ (uint32_t)(uMax - uMin + 31) / 32, (uint32_t)(vMax - vMin + 31) / 32, 1 };

				kernelRepeaterUVLoop<float, SIZE_GLOBAL, SIZE_REPEATER, exchange> << < dimGrid, dimBlock >> > (
					target.gpuGetPtr(),
					target.getWidth(),
					target.getHeight(),

					repeater.gpuGetPtr(),
					repeater.getWidth(),
					repeater.getHeight(),

					u,
					v,

					uMin,
					vMin
					);

				cudaDeviceSynchronize();
			}
		};
	}
}
#pragma once
/**
*	Author: Norman Pellet
*	Date: April 2nd, 2020
*	UV repeater code executed on the GPU. Looping over target grid
*/
#include "../test_repeater.hpp"
#include "../../../util/gpuutil.cuh"
#include <stdint.h>

namespace Tests {
	namespace RepeaterGeneral {
		template<typename T, uint32_t SIZE_GLOBAL, uint32_t SIZE_REPEATER>
		void __global__ kernelRepeaterNonSquare(
			T* target,
			const uint32_t width,
			const uint32_t height,

			const T* repeater,
			const uint32_t repeaterWidth,
			const uint32_t repeaterHeight,

			const int2 u,
			const int2 v,

			const float2 uNorm,
			const float2 vNorm,

			int32_t uMin,
			int32_t uMax,
			int32_t vMin,
			int32_t vMax
			) {
			const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
			const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
			if (x >= width || y >= height) { return; }
			const int32_t uComp = (int32_t)(dot(uNorm, 1, x, y)); // Integer clamp
			const int32_t vComp = (int32_t)(dot(vNorm, 1, x, y));

			float val = 0;
			int16_t num = 0;

			// If the repeater size if larger than the repetition vector, there is some blending
			for (int32_t uR = uMin; uR <= uMax; uR++) {
				for (int32_t vR = vMin; vR <= vMax; vR++) {
					const int32_t startX = (uComp + uR) * u.x + (vComp + vR) * v.x;
					const int32_t xRep = x - startX;
					const int32_t startY = (uComp + uR) * u.y + (vComp + vR) * v.y;
					const int32_t yRep = y - startY;
					// Boundary check
					if (xRep < 0 || yRep < 0 || xRep >= SIZE_REPEATER || yRep >= SIZE_REPEATER) {
						continue;
					}

					const float locval = repeater[getMatrixIndex<size_t>(xRep, yRep, SIZE_REPEATER)];
					if (locval != 0) {
						val += locval;
						num++;
					}
				}
			}
			if (num > 0) {
				target[getMatrixIndex<size_t>(x, y, width)] = val / num;
			}
		}

		template<uint32_t SIZE_REPEATER, uint32_t SIZE_GLOBAL, bool exchange>
		class GPU : public Base<SIZE_REPEATER, SIZE_GLOBAL> {
		public:

			GPU(const std::string descr, const uint32_t num) : Base<SIZE_REPEATER, SIZE_GLOBAL>(descr, num) {}

			void testSetup() {
				Base<SIZE_REPEATER, SIZE_GLOBAL>::testSetup();
				target.fillWith(0);
				target.gpuAllocate(true, true);
				repeater.gpuAllocate();
			}

			void testExecute() {
				assert(uMax >= uMin);
				assert(vMax >= vMin);

				kernelRepeaterNonSquare<float, SIZE_GLOBAL, SIZE_REPEATER> << < target.dimGrid, target.dimBlock >> > (
					target.gpuGetPtr(),
					target.getWidth(),
					target.getHeight(),

					repeater.gpuGetPtr(),
					repeater.getWidth(),
					repeater.getHeight(),

					u,
					v,

					uNorm,
					vNorm,

					uMin,
					uMax,
					vMin,
					vMax
					);

				cudaDeviceSynchronize();
			}
		};
	}
}
#pragma once
/**
*	Author: Norman Pellet
*	Date: April 2nd, 2020
*	Distance kernel with that calculates d = dx^2 + dy^2, where dx and dy are offset coords
*	to the nearest white pixel.
*/
#include <stdint.h>

#include "../test_distance.hpp"
#include "../../../util/gpuutil.cuh"

namespace Tests {
	namespace Distance {
		template<typename T, typename U>
		__global__ void kernelDistanceBlack(
			const T* source,
			const uint32_t width,
			const uint32_t height,

			U* distance,
			const int32_t distanceMax
			) {
			const int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
			const int32_t row = blockIdx.y * blockDim.y + threadIdx.y;

			if (col >= width || row >= height) {
				return;
			}

			const size_t index = getMatrixIndex<size_t>(col, row, width);

			if (source[index] != 0) {
				distance[index] = 0;
				return;
			}

			U minDistance = distanceMax * distanceMax;

			for (int32_t y = -distanceMax; y <= distanceMax; y++) {
				for (int32_t x = -distanceMax; x <= distanceMax; x++) {
					const U dist = (U)(x * x + y * y);
					if (dist > minDistance) {
						continue;
					}

					const int32_t col2 = col + x;
					const int32_t row2 = row + y;

					if (col2 < 0 || row2 < 0 || col2 >= width || row2 >= height) {
						continue;
					}
					const size_t index2 = getMatrixIndex<size_t>(col2, row2, width);

					if (source[index2] != 0) {
						minDistance = min(minDistance, dist);
					}
				}
			}

			distance[index] = minDistance;
		}

		template<uint32_t maxDistance>
		class GPUDistanceBlack : public Base {
		private:
		public:

			GPUDistanceBlack(const std::string source, const std::string descr, const uint32_t num) : Base(source, descr, num) {}

		protected:

			void testSetup() override {
				Base::testSetup();
				distance->fillWith(maxDistance * maxDistance);

				source->gpuAllocate();
				distance->gpuAllocate();
			};

			void testExecute() {
				kernelDistanceBlack< unsigned char, unsigned int> << < source->dimGrid, source->dimBlock >> > (
					source->gpuGetPtr(),
					source->getWidth(),
					source->getHeight(),

					distance->gpuGetPtr(),
					maxDistance
					);
				checkCudaKernel();
				cudaDeviceSynchronize();
			}
		};
	}
}
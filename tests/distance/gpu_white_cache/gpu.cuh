#pragma once
/**
*	Author: Norman Pellet
*	Date: April 2nd, 2020
*	Distance kernel with that calculates d = dx^2 + dy^2 for surrounding black pixels
*	and keeps the lowest distance value. Distance is cached into a matrix
*/

#include <stdint.h>

#include "../test_distance.hpp"
#include "../../../util/gpuutil.cuh"

namespace Tests {
	namespace Distance {
		template<typename T, typename U, bool neighbourCheck>
		__global__ void kernelDistanceCache(
			const T* source,
			const uint32_t width,
			const uint32_t height,

			U* distance,
			const int32_t distanceMax,
			const U* distanceCache
			) {
			const int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
			const int32_t row = blockIdx.y * blockDim.y + threadIdx.y;

			if (col >= width || row >= height) {
				return;
			}

			const size_t index = getMatrixIndex<size_t>(col, row, width);

			if (source[index] == 0) {
				return;
			}

			if (neighbourCheck) {
				// If we're inside the boundaries and surrounded by 4 non-black pixels, then no need to execute the matrix patching
				if (col > 0 && col < width - 1 && row > 0 && row < height - 1 && source[index - 1] > 0 && source[index + 1] > 0 && source[index - width] > 0 && source[index + width] > 0) {
					distance[index] = 0;
					return;
				}
			}

			for (int32_t y = -distanceMax; y < distanceMax; y++) {
				for (int32_t x = -distanceMax; x < distanceMax; x++) {
					const int32_t colDist = col + x;
					const int32_t rowDist = row + y;

					if (colDist < 0 || rowDist < 0 || colDist >= width || rowDist >= height) {
						continue;
					}
					const size_t indexDistance = getMatrixIndex<size_t>(colDist, rowDist, width);

					atomicMin((U*)(distance + indexDistance), distanceCache[getMatrixIndex(x + distanceMax, y + distanceMax, distanceMax * 2 + 1)]);
				}
			}
		}

		template<typename T, typename U, uint32_t maxDistance, bool neighbourCheck>
		__global__ void kernelDistanceCacheShared(
			const T* source,
			const uint32_t width,
			const uint32_t height,

			U* distance,
			const int32_t distanceMax,
			const U* distanceCache
			) {
			const uint32_t distMatrixSize = maxDistance * 2 + 1;
			const int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
			const int32_t row = blockIdx.y * blockDim.y + threadIdx.y;

			__shared__ U distanceCacheShared[distMatrixSize * distMatrixSize];

			if (threadIdx.x < distMatrixSize && threadIdx.y < distMatrixSize) {
				distanceCacheShared[threadIdx.y * distMatrixSize + threadIdx.x] = pow2((int32_t)threadIdx.x - maxDistance - 1) + pow2((int32_t)threadIdx.y - maxDistance - 1);
			}

			__syncthreads();

			if (col >= width || row >= height) {
				return;
			}

			const size_t index = getMatrixIndex<size_t >(col, row, width);

			if (source[index] == 0) {
				return;
			}

			if (neighbourCheck) {
				// If we're inside the boundaries and surrounded by 4 non-black pixels, then no need to execute the matrix patching
				if (col > 0 && col < width - 1 && row > 0 && row < height - 1 && source[index - 1] > 0 && source[index + 1] > 0 && source[index - width] > 0 && source[index + width] > 0) {
					distance[index] = 0;
					return;
				}
			}

			for (int32_t y = -distanceMax; y < distanceMax; y++) {
				for (int32_t x = -distanceMax; x < distanceMax; x++) {
					const int32_t colDist = col + x;
					const int32_t rowDist = row + y;

					if (colDist < 0 || rowDist < 0 || colDist >= width || rowDist >= height) {
						continue;
					}
					const size_t indexDistance = getMatrixIndex<size_t>(colDist, rowDist, width);

					atomicMin((U*)(distance + indexDistance), (U)(x * x + y * y));
				}
			}
		}

		template<uint32_t maxDistance, bool neighbourCheck>
		class GPUDistanceWhiteCache : public Base {
		private:
			dim3 dimBlock;
			dim3 dimGrid;
			typedTile<uint32_t, GPUMatrixAllocator<uint32_t>> cache{ maxDistance * 2 + 1, maxDistance * 2 + 1 };

		public:

			GPUDistanceWhiteCache(const std::string source, const std::string descr, const uint32_t num) : Base(source, descr, num) {}

		protected:

			void testSetup() override {
				Base::testSetup();
				distance->fillWith(maxDistance * maxDistance);

				source->gpuAllocate();
				distance->gpuAllocate();

				for (int x = -maxDistance; x < maxDistance; x++) {
					for (int y = -maxDistance; y < maxDistance; y++) {
						cache.loadPointXY(x + maxDistance, y + maxDistance, x * x + y * y);
					}
				}

				cache.gpuAllocate();
			};

			void testExecute() {
				kernelDistanceCacheShared< unsigned char, unsigned int, maxDistance, neighbourCheck > << < source->dimGrid, source->dimBlock >> > (
					source->gpuGetPtr(),
					source->getWidth(),
					source->getHeight(),

					distance->gpuGetPtr(),
					maxDistance,

					cache.gpuGetPtr()
					);
				checkCudaKernel();
				cudaDeviceSynchronize();
			}
		};
	}
}
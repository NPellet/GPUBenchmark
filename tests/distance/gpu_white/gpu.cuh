#pragma once

/**
*	Author: Norman Pellet
*	Date: April 2nd, 2020
*	Distance kernel with that calculates d = dx^2 + dy^2 for surrounding black pixels
*	and keeps the lowest distance value.
*/
#include "../../../util/gpuutil.cuh"
#include "../test_distance.hpp"
#include <stdint.h>

namespace Tests {
	namespace Distance {
		template<typename T, typename U, bool neighbourCheck, bool distanceCheck>
		__global__ void kernelDistance(
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

			for (int32_t y = -distanceMax; y <= distanceMax; y++) {
				for (int32_t x = -distanceMax; x <= distanceMax; x++) {
					const int32_t colDist = col + x;
					const int32_t rowDist = row + y;

					if (colDist < 0 || rowDist < 0 || colDist >= width || rowDist >= height) {
						continue;
					}
					const size_t indexDistance = getMatrixIndex<size_t>(colDist, rowDist, width);
					const U distVal = (U)(x * x + y * y);
					if (distanceCheck) {
						if (distance[indexDistance] < distVal) {
							continue;
						}
					}
					atomicMin((U*)(distance + indexDistance), distVal);
				}
			}
		}
		template<typename T, typename U>
		__global__ void kernelDistance_ncheck(
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

			if (source[index] == 0) {
				return;
			}

			int32_t xFrom = -distanceMax, xTo = distanceMax, yFrom = -distanceMax, yTo = distanceMax;
			if (col == 0 || source[index - 1] != 0x00) 	xFrom = 0;
			if (col == width - 1 || source[index + 1] != 0x00) 	xTo = 0;
			if (row == 0 || source[index - width] != 0x00) yFrom = 0;
			if (row == height - 1 || source[index + width] != 0x00)	yTo = 0;

			for (int32_t y = yFrom; y <= yTo; y++) {
				for (int32_t x = xFrom; x <= xTo; x++) {
					const int32_t colDist = col + x;
					const int32_t rowDist = row + y;

					if (colDist < 0 || rowDist < 0 || colDist >= width || rowDist >= height) {
						continue;
					}

					const size_t indexDistance = getMatrixIndex<size_t>(colDist, rowDist, width);
					const U distVal = (U)(x * x + y * y);
					atomicMin((U*)(distance + indexDistance), distVal);
				}
			}
		}

		template<uint32_t maxDistance, bool neighbourCheck, bool spreadBlock, bool distanceCheck>
		class GPUDistanceWhite : public Base {
		private:
			dim3 dimBlock;
			dim3 dimGrid;
		public:

			GPUDistanceWhite(const std::string source, const std::string descr, const uint32_t num) : Base(source, descr, num) {}

			void testExecute_neighbourTest() {
				kernelDistance_ncheck< unsigned char, unsigned int > << < dimGrid, dimBlock >> > (
					source->gpuGetPtr(),
					source->getWidth(),
					source->getHeight(),

					distance->gpuGetPtr(),
					maxDistance
					);
				checkCudaKernel();
				cudaDeviceSynchronize();
			}
		protected:

			void testSetup() override {
				Base::testSetup();
				distance->fillWith(maxDistance * maxDistance);

				source->gpuAllocate();
				distance->gpuAllocate();

				if (spreadBlock) {
					dimBlock = { 1024, 1, 1 };
					dimGrid = source->calcDimGrid(dimBlock);
				}
				else {
					dimBlock = source->dimBlock;
					dimGrid = source->dimGrid;
				}
			};

			void testExecute() {
				kernelDistance< unsigned char, unsigned int, neighbourCheck, distanceCheck > << < dimGrid, dimBlock >> > (
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
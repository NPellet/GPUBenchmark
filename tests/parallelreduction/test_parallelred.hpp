#pragma once

#include <stdint.h>
#include "../../test/test.hpp"
#include "../../tiles/typedTile.hpp"
#include "../../tiles/allocators/gpumatrix.cuh"
#include "../../util/tiffload.hpp"
#include "../../util/gpuutil.cuh"

namespace Tests {
	namespace ParallelReduction {
		template<typename T, typename U>
		void __global__ kernelRescaleZ(
			U* source,
			const uint32_t w,
			const uint32_t h,
			T* dest,
			const U minIn,
			const U maxIn,

			const T minOut,
			const T maxOut
			) {
			const int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
			const int32_t row = blockIdx.y * blockDim.y + threadIdx.y;

			if (col >= w || row >= h) {
				return;
			}

			const size_t index = getMatrixIndex<size_t>(col, row, w);
			dest[index] = ((T)source[index] - minIn) / (maxIn - minIn) * (maxOut - minOut) + minOut;
		}

		class Base : public Test {
		protected:

			using zType = unsigned int;
			using Tile = typedTile<unsigned char, GPUMatrixAllocator<unsigned char>>;
			using TileZScaled = typedTile<zType, GPUMatrixAllocator<zType>>;

			Tile* source;
			TileZScaled* source_rescaled;

		public:

			Base(const std::string filePath, const std::string descr, uint32_t numRep) : Test(descr, numRep) {
				source = loadTiff<Tile>(filePath.c_str());
				source->gpuAllocate();
				source_rescaled = new TileZScaled(*source);
				source_rescaled->gpuAllocate();

				kernelRescaleZ<zType, unsigned char> << < source->dimGrid, source->dimBlock >> > (
					source->gpuGetPtr(),
					source->getWidth(),
					source->getHeight(),
					source_rescaled->gpuGetPtr(),
					0,
					255,
					0,
					10.0
					);

				checkCudaKernel();
			}

			~Base() {
			}

		protected:

			// Tiles are allocated on the stack, so no need to do memory management in the test setup and destroy
			void testSetup() override {
				source_rescaled->gpuAllocate();
			};

			void testDestroy() override {
				delete source;
				delete source_rescaled;
			};
		};
	}
}
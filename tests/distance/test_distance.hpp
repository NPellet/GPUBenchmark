#pragma once
#include <stdint.h>
#include <iostream>

#include "../../test/test.hpp"
#include "../../tiles/typedTile.hpp"
#include "../../tiles/allocators/gpumatrix.cuh"
#include "../../util/tiffload.hpp"

namespace Tests {
	namespace Distance {
		class Base : public Test {
		public:

			using Tile = typedTile<unsigned char, GPUMatrixAllocator<unsigned char>>;
			using TileDistance = typedTile<unsigned int, GPUMatrixAllocator<unsigned int>>;

			Base(const std::string filePath, const std::string descr, uint32_t numRep) : Test(descr, numRep) {
				source = loadTiff<Tile>(filePath.c_str());
				distance = new TileDistance{ source->width, source->height };
			}

			~Base() {
			}

		protected:
			Tile* source;
			TileDistance* distance;

			// Tiles are allocated on the stack, so no need to do memory management in the test setup and destroy
			void testSetup() override {
			};

			void testDestroy() override {
				distance->gpuRetrieve();
				verificationValue = CRC32(distance->getPtr(), static_cast<size_t>(distance->width) * distance->height);
				delete source;
				delete distance;
			};
		};
	}
}
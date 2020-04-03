#pragma once

#include <stdint.h>
#include "../../test/test.hpp"
#include "../../tiles/typedTile.hpp"
#include "../../tiles/allocators/gpumatrix.cuh"

namespace Tests {
	namespace Repeater {
		template<uint32_t SIZE_REPEATER, uint32_t SIZE_GLOBAL>
		class Base : public Test {
		public:
			Base(const std::string descr, uint32_t numRep) : Test(descr, numRep) {}

			~Base() {
			}

		protected:
			typedTile<float, GPUMatrixAllocator<float>> repeater{ SIZE_REPEATER, SIZE_REPEATER };
			typedTile<float, GPUMatrixAllocator<float>> target{ SIZE_GLOBAL, SIZE_GLOBAL };

			// Tiles are allocated on the stack, so no need to do memory management in the test setup and destroy
			void testSetup() override {
				const double radius = SIZE_REPEATER / 2.0;

				// However we are going to fill the repeater with some data
				for (int x = 0; x < SIZE_REPEATER; x++) {
					for (int y = 0; y < SIZE_REPEATER; y++) {
						double dist = std::hypot(x - SIZE_REPEATER / 2.0, y - SIZE_REPEATER / 2.0); // Cannot inline with the condition, because we're compiling in C++14. CUDA limitation...
						if (dist < radius) {
							repeater.loadPointXY(x, y, (float)pow(dist, 2));
						}
					}
				}
			};

			void testDestroy() override {
				target.gpuRetrieve();
				CRCValue = CRC32(target.getPtr(), SIZE_GLOBAL * SIZE_GLOBAL);
			};
		};
	}
}
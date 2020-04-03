#pragma once
/**
*	Author: Norman Pellet
*	Date: April 2nd, 2020
*	UV repeater code executed on the host. Looping over the UV coords
*/
#include "../test_repeater.hpp"
#include "../../../util/gpuutil.cuh"
#include <stdint.h>

namespace Tests {
	namespace RepeaterGeneral {
		template<uint32_t SIZE_REPEATER, uint32_t SIZE_GLOBAL>
		class HostUV : public Base<SIZE_REPEATER, SIZE_GLOBAL> {
		public:

			HostUV(const std::string descr, const uint32_t num) : Base<SIZE_REPEATER, SIZE_GLOBAL>(descr, num) {}

			void testExecute() {
				updateBoundaries(0, 0);
				updateBoundaries(SIZE_GLOBAL, 0);
				updateBoundaries(0, SIZE_GLOBAL);
				updateBoundaries(SIZE_GLOBAL, SIZE_GLOBAL);

				for (int32_t uGrid = uMin; uGrid <= uMax; uGrid++) {
					for (int32_t vGrid = vMin; vGrid <= vMax; vGrid++) {
						const int32_t startX = uGrid * u.x + vGrid * v.x;
						const int32_t startY = uGrid * u.y + vGrid * v.y;

						size_t repeaterIndex = 0;
						for (uint32_t x = 0; x < SIZE_REPEATER; x++) {
							for (uint32_t y = 0; y < SIZE_REPEATER; y++) {
								if (startX + x < 0 || startY + y < 0 || startX + x >= SIZE_GLOBAL || startY + y >= SIZE_GLOBAL) {
									repeaterIndex++;
									continue;
								}

								const size_t index = target.calcIndex(startX + x, startY + y);
								target.loadPoint(index, repeater.getPoint(repeaterIndex));
								repeaterIndex++;
							}
						}
					}
				}
			}
		};
	}
}
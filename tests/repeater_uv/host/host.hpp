#pragma once
/**
*	Author: Norman Pellet
*	Date: April 2nd, 2020
*	UV repeater code executed on the host. Looping over the target grid
*/
#include "../test_repeater.hpp"
#include "../../../util/gpuutil.cuh"
#include <stdint.h>

namespace Tests {
	namespace RepeaterGeneral {
		template<uint32_t SIZE_REPEATER, uint32_t SIZE_GLOBAL>
		class Host : public Base<SIZE_REPEATER, SIZE_GLOBAL> {
		public:

			Host(const std::string descr, const uint32_t num) : Base<SIZE_REPEATER, SIZE_GLOBAL>(descr, num) {}

			void testExecute() {
				size_t index = 0; // Assumes we know how the data is placed
				for (uint32_t y = 0; y < SIZE_GLOBAL; y++) {
					for (uint32_t x = 0; x < SIZE_GLOBAL; x++) {
						// Dot product over the normalized repetition vector
						const int32_t uComp = (int32_t)(dot(uNorm, 1, x, y));
						const int32_t vComp = (int32_t)(dot(vNorm, 1, x, y));

						// Accumulation
						float val = 0;
						// Number of samples for blending
						int16_t num = 0;

						// There might be more than one (u,v) to check for this position
						for (int32_t uR = uMin; uR <= uMax; uR++) {
							for (int32_t vR = vMin; vR <= vMax; vR++) {
								const int32_t startX = (uComp + uR) * u.x + (vComp + vR) * v.x;
								const int32_t xRep = x - startX;

								const int32_t startY = (uComp + uR) * u.y + (vComp + vR) * v.y;
								const int32_t yRep = y - startY;

								if (xRep < 0 || yRep < 0 || xRep >= SIZE_REPEATER || yRep >= SIZE_REPEATER) {
									continue;
								}

								float locval = repeater.getPointXY(xRep, yRep);
								if (locval != 0) {
									val += locval;
									num++;
								}
							}
						}
						if (num > 0) {
							target.loadPoint(index, val / num);
						}

						index++;
					}
				}
			}
		};
	}
}
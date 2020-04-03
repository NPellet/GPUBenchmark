#pragma once
/**
*	Author: Norman Pellet
*	Date: April 2nd, 2020
*	Simple repeater code executed on the host. Looping over the repeater
*/
#include "../test_repeater.hpp"
#include <stdint.h>

namespace Tests {
	namespace Repeater {
		template<uint32_t SIZE_REPEATER, uint32_t SIZE_GLOBAL>
		class HostLoopRepeater : public Base<SIZE_REPEATER, SIZE_GLOBAL> {
		public:

			HostLoopRepeater(const std::string descr, const uint32_t num) : Base<SIZE_REPEATER, SIZE_GLOBAL>(descr, num) {}

		protected:

			void testExecute() {
				size_t repeaterIndex = 0;
				for (uint32_t x = 0; x < SIZE_REPEATER; x++) {
					for (uint32_t y = 0; y < SIZE_REPEATER; y++) {
						const float repeaterValue = repeater.getPoint(repeaterIndex);

						for (uint32_t xB = x; xB < SIZE_GLOBAL; xB += SIZE_REPEATER) {
							for (uint32_t yB = y; yB < SIZE_GLOBAL; yB += SIZE_REPEATER) {
								target.loadPointXY(xB, yB, repeaterValue);
							}
						}

						repeaterIndex++;
					}
				}
			}
		};
	}
}
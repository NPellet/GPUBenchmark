#pragma once
/**
*	Author: Norman Pellet
*	Date: April 2nd, 2020
*	Simple repeater code executed on the host. Looping over target grid
*/
#include "../test_repeater.hpp"
#include <iostream>
#include <stdint.h>

namespace Tests {
	namespace Repeater {
		template<uint32_t SIZE_REPEATER, uint32_t SIZE_GLOBAL>
		class Host : public Base<SIZE_REPEATER, SIZE_GLOBAL> {
		public:

			Host(const std::string descr, const uint32_t num) : Base<SIZE_REPEATER, SIZE_GLOBAL>(descr, num) {}

			void testExecute() {
				for (uint32_t x = 0; x < SIZE_GLOBAL; x++) {
					for (uint32_t y = 0; y < SIZE_GLOBAL; y++) {
						target.loadPointXY_assert(x, y, repeater.getPointXY(x % SIZE_REPEATER, y % SIZE_REPEATER));
					}
				}
			}

			void testExecute_indexed() {
				size_t index = 0;
				for (uint32_t y = 0; y < SIZE_GLOBAL; y++) {
					for (uint32_t x = 0; x < SIZE_GLOBAL; x++) {
						target.loadPoint(index, repeater.getPointXY(x % SIZE_REPEATER, y % SIZE_REPEATER));
						index++;
					}
				}
			}

			void testExecute_indexed_incrPointer() {
				float* ptr = target.getPtr();
				for (uint32_t y = 0; y < SIZE_GLOBAL; y++) {
					for (uint32_t x = 0; x < SIZE_GLOBAL; x++) {
						*ptr = repeater.getPointXY(x % SIZE_REPEATER, y % SIZE_REPEATER);
						ptr++;
					}
				}
			}

		protected:
		};
	}
}
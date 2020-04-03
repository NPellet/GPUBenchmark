#pragma once
#include <cuda_runtime.h>
#include <utility>
#include "../repeater/test_repeater.hpp"

namespace Tests {
	namespace RepeaterGeneral {
		template<uint32_t SIZE_REPEATER, uint32_t SIZE_GLOBAL>
		class Base : public ::Tests::Repeater::Base<SIZE_REPEATER, SIZE_GLOBAL> {
		public:
			using BaseClass = ::Tests::Repeater::Base<SIZE_REPEATER, SIZE_GLOBAL>;
			Base(const std::string descr, uint32_t numRep) : BaseClass(descr, numRep) {}

			void setRepetition(int2 rep1, int2 rep2) {
				u = rep1;
				v = rep2;

				_rep1l = mag2(u);
				_rep2l = mag2(v);

				uNorm.x = u.x / _rep1l;
				uNorm.y = u.y / _rep1l;
				vNorm.x = v.x / _rep2l;
				vNorm.y = v.y / _rep2l;

				updateBoundaries(0, 0);
				updateBoundaries(SIZE_REPEATER, 0);
				updateBoundaries(0, SIZE_REPEATER);
				updateBoundaries(SIZE_REPEATER, SIZE_REPEATER);
			}

		private:

		protected:
			void updateBoundaries(int32_t x, int32_t y) {
				uMin = min(uMin, (int32_t)floor(dot(u, _rep1l, x, y)));
				uMax = max(uMax, (int32_t)ceil(dot(u, _rep1l, x, y)));

				vMin = min(vMin, (int32_t)floor(dot(v, _rep2l, x, y))); // Flooring is important, when the result is negative
				vMax = max(vMax, (int32_t)ceil(dot(v, _rep2l, x, y)));
			}
			int32_t uMin{}, uMax{}, vMin{}, vMax{};
			int2 u, v; // Repetition vectors
			float2 uNorm, vNorm; // Normalized repetition vectors
			float _rep1l, _rep2l;
		};
	}
}
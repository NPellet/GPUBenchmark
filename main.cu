#include "test/test.hpp"

#include "tests/repeater/benchmark_repeater.cuh"
#include "tests/repeater_uv/benchmark_repeater_uv.cuh"

// Will not compile due to missing tiff reading capabilities. I cannot include that file publicly :(
#include "tests/distance/benchmark_distance.cuh"

int main() {
	benchmark_repeater();
	benchmark_repeater_uv();

	// Will not compile due to missing tiff reading capabilities.
	benchmark_distance();
}
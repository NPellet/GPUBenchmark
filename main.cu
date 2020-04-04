#include "test/test.hpp"

#include "tests/parallelreduction/benchmark_parallelred.cuh"
#include "tests/repeater/benchmark_repeater.cuh"
#include "tests/repeater_uv/benchmark_repeater_uv.cuh"
#include "tests/distance/benchmark_distance.cuh"

int main() {
	benchmark_parallelreduction();

	benchmark_repeater();
	benchmark_repeater_uv();
	benchmark_distance();
}
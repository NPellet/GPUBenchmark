#include "parallelreduction/parallelreduction.cuh"
#include "atomics/atomics.cuh"
#include "benchmark_parallelred.cuh"

void benchmark_parallelreduction() {
	Test::init("report_parallelred.txt");

	namespace nsTest = Tests::ParallelReduction;
	const std::string fileName = "./source_distance.tif";

	[&fileName]() {
		nsTest::GPUAtomics test{ fileName,  "Simple atomics", 1000 };
		test.execute();
	}();

	[&fileName]() {
		using rep = nsTest::GPUAtomics;
		rep test{ fileName, "Simple atomics with shared memory", 1000 };
		test.execute(std::bind(&rep::testExecute_smem, &test));
	}();

	[&fileName]() {
		nsTest::GPUParallelReduction test{ fileName,  "Parallel reduction", 1000 };
		test.execute();
	}();

	[&fileName]() {
		using rep = nsTest::GPUParallelReduction;
		rep test{ fileName, "Parallel reduction with 2x loads", 1000 };
		test.execute(std::bind(&rep::testExecute_multLoad<2>, &test));
	}();

	[&fileName]() {
		using rep = nsTest::GPUParallelReduction;
		rep test{ fileName, "Parallel reduction with 2x loads loads and unrolling", 1000 };
		test.execute(std::bind(&rep::testExecute_multLoad_unroll<2>, &test));
	}();

	[&fileName]() {
		using rep = nsTest::GPUParallelReduction;
		rep test{ fileName, "Parallel reduction with 4x loads and unrolling", 1000 };
		test.execute(std::bind(&rep::testExecute_multLoad_unroll<4>, &test));
	}();

	[&fileName]() {
		using rep = nsTest::GPUParallelReduction;
		rep test{ fileName, "Parallel reduction with 8x loads and unrolling", 1000 };
		test.execute(std::bind(&rep::testExecute_multLoad_unroll<8>, &test));
	}();

	[&fileName]() {
		using rep = nsTest::GPUParallelReduction;
		rep test{ fileName, "Parallel reduction with 16x loads and unrolling", 1000 };
		test.execute(std::bind(&rep::testExecute_multLoad_unroll<16>, &test));
	}();

	Test::end();
}
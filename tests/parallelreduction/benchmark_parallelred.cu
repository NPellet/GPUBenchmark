#include "atomics/atomics.cuh"
#include "benchmark_parallelred.cuh"

void benchmark_parallelreduction() {
	Test::init("report_parallelred.txt");

	namespace nsTest = Tests::ParallelReduction;

	constexpr uint32_t distanceMax = 15;
	const std::string fileName = "./source_distance.tif";

	[&fileName]() {
		nsTest::GPUAtomics test{ fileName,  "Simple atomics", 10 };
		test.execute();
	}();

	[&fileName]() {
		using rep = nsTest::GPUAtomics;
		rep test{ fileName, "Simple atomics with shared memory", 10 };
		test.execute(std::bind(&rep::testExecute_smem, &test));
	}();
	Test::end();
}
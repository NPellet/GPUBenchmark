#include "benchmark_repeater.cuh"

#include "host/host.hpp"
#include "host_loopOverRepeater/host_loopOverRepeater.hpp"
#include "gpu/gpu.cuh"
#include "gpu_shared/gpu_shared.cuh"
#include "gpu_repeaterGrid/gpu_repeaterGrid.cuh"

void benchmark_repeater() {
	Test::init("report_repeater.txt");

	namespace nsTest = Tests::Repeater;

	constexpr uint32_t repeaterSize = 100;
	constexpr uint32_t size = 10240;

	[]() {
		nsTest::Host< repeaterSize, size > test{ "Host repetition with loop over target", 10 };
		test.execute();
	}();

	[]() {
		using rep = nsTest::Host< repeaterSize, size>;
		rep test{ "Host repetition with loop over target, indexed", 10 };
		test.execute(std::bind(&rep::testExecute_indexed, &test));
	}();

	[]() {
		using rep = nsTest::Host< repeaterSize, size>;
		rep test{ "Host repetition with loop over target, direct to pointer", 10 };
		test.execute(std::bind(&rep::testExecute_indexed_incrPointer, &test));
	}();

	[]() {
		nsTest::HostLoopRepeater< repeaterSize, size > test{ "Host repetition with loop over repeater", 10 };
		test.execute();
	}();

	[]() {
		nsTest::GPURepeater< repeaterSize, size, true, size_t > test{ "GPU repetition with grid over target - including copy", 100 };
		test.execute();
	}();

	[]() {
		nsTest::GPURepeater< repeaterSize, size, false, size_t > test{ "GPU repetition with grid over target", 100 };
		test.execute();
	}();

	[]() {
		nsTest::GPURepeater< repeaterSize, size, false, uint32_t > test{ "GPU repetition with grid over target - uint32", 100 };
		test.execute();
	}();

	[]() {
		nsTest::GPURepeaterShared< repeaterSize, size > test{ "GPU repetition with grid over target using shared mem", 100 };
		test.execute();
	}();

	[]() {
		nsTest::GPURepeaterRepGrid< repeaterSize, size > test{ "GPU repetition with grid over repeater", 100 };
		test.execute();
	}();

	Test::end();
}
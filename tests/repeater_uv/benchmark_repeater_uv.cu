#include "benchmark_repeater_uv.cuh"

void benchmark_repeater_uv() {
	namespace nsTest = Tests::RepeaterGeneral;

	constexpr uint32_t repeaterSize = 100;
	constexpr uint32_t size = 8000;

	Test::init("report_repeater_uv.txt");

	[]() {
		using rep = nsTest::Host< repeaterSize, size>;
		rep test{ "Host repetition with loop over target, indexed", 1 };
		test.setRepetition({ 110, 110 }, { 110, -110 });
		test.execute();
	}();
	[]() {
		using rep = nsTest::HostUV< repeaterSize, size>;
		rep test{ "Host repetition with loop over UV coords", 1 };
		test.setRepetition({ 110, 110 }, { 110, -110 });
		test.execute();
	}();

	[]() {
		using rep = nsTest::GPU< repeaterSize, size, false>;
		rep test{ "GPU repetition with loop over target", 10 }; // Can only do once, as we're using an atomicAdd
		test.setRepetition({ 110, 110 }, { 110, -110 });
		test.execute();
	}();

	[]() {
		using rep = nsTest::GPUUV< repeaterSize, size, false>;
		rep test{ "GPU repetition with loop over UV coords - reassign value", 10 }; // Can only do once, as we're using an atomicAdd
		test.setRepetition({ 110, 110 }, { 110, -110 });
		test.execute();
	}();

	[]() {
		using rep = nsTest::GPUUV< repeaterSize, size, true>;
		rep test{ "GPU repetition with loop over UV coords - atomic exchange", 10 }; // Can only do once, as we're using an atomicAdd
		test.setRepetition({ 110, 110 }, { 110, -110 });
		test.execute();
	}();

	Test::end();
}
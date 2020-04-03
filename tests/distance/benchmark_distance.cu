#include "gpu_white/gpu.cuh"
#include "gpu_black/gpu.cuh"
#include "gpu_white_cache/gpu.cuh"

void benchmark_distance() {
	Test::init("report_distance.txt");

	namespace nsTest = Tests::Distance;

	constexpr uint32_t distanceMax = 15;
	const std::string fileName = "./source_distance.tif";

	[&fileName]() {
		nsTest::GPUDistanceWhite< distanceMax, false, false, false> test{ fileName,  "look for white - atomicMin on black", 10 };
		test.execute();
	}();

	[&fileName]() {
		nsTest::GPUDistanceWhite< distanceMax, false, true, false> test{ fileName,  "look for white - atomicMin on black - block spreading", 10 };
		test.execute();
	}();

	[&fileName]() {
		nsTest::GPUDistanceWhite< distanceMax, true, false, false > test{ fileName,  "look for white - atomicMin on black - neighbour check", 10 };
		test.execute();
	}();

	[&fileName]() {
		nsTest::GPUDistanceWhite< distanceMax, false, false, true > test{ fileName,  "look for white - distance check", 10 };
		test.execute();
	}();

	[&fileName]() {
		nsTest::GPUDistanceWhite< distanceMax, false, true, true > test{ fileName,  "look for white - distance check - block spread", 10 };
		test.execute();
	}();

	[&fileName]() {
		nsTest::GPUDistanceWhite< distanceMax, true, false, true > test{ fileName,  "look for white - neighbour + distance check", 10 };
		test.execute();
	}();

	[&fileName]() {
		using testClass = nsTest::GPUDistanceWhite< distanceMax, true, false, true >;
		testClass test{ fileName, "look for white - loop restriction", 10 };
		test.execute(std::bind(&testClass::testExecute_neighbourTest, &test));
	}();

	[&fileName]() {
		nsTest::GPUDistanceWhiteCache< distanceMax, true > test{ fileName,  "look for white - neighbour check + cached distance kernel", 10 };
		test.execute();
	}();

	[&fileName]() {
		nsTest::GPUDistanceBlack< distanceMax > test{ fileName,  "Black", 10 };
		test.execute();
	}();

	Test::end();
}
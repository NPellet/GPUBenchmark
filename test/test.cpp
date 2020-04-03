#include "test.hpp"

#include <chrono>
#include <string>
#include <iostream>
#include <fstream>

Test::Test(const std::string descr, const uint32_t num) : description(descr), numberOfIterations(num) { };

void Test::init(std::string fileName) {
	reportStream.open(fileName);

	if (!reportStream.is_open()) {
		std::cerr << "Could not open report file: " << fileName << std::endl;
		exit(EXIT_FAILURE);
	}
}

void Test::end() {
	reportStream.close();
}

void Test::report() {
	reportStream << "Reporting test \"" << description << "\" with " << std::to_string(numberOfIterations) << " iterations: " << std::to_string((stop - start).count() / 1000 / numberOfIterations) << " us / iteration" << std::endl;
	reportStream << "\t\tWidth CRC value " << std::hex << CRCValue << std::endl;
}

void Test::testStart() {
	start = std::chrono::high_resolution_clock::now();
}

void Test::testEnd() {
	stop = std::chrono::high_resolution_clock::now();
}

void Test::execute() {
	testSetup();

	testStart();
	for (int i = 0; i < numberOfIterations; i++) {
		testExecute();
	}

	testEnd();
	testDestroy();

	report();
}

void Test::execute(std::function<void()> func) {
	testSetup();

	testStart();
	for (int i = 0; i < numberOfIterations; i++) {
		func();
	}

	testEnd();
	testDestroy();

	report();
}

std::ofstream Test::reportStream;
std::chrono::time_point< std::chrono::high_resolution_clock > Test::start;
std::chrono::time_point< std::chrono::high_resolution_clock > Test::stop;
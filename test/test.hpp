#pragma once

#include <chrono>
#include <string>
#include <functional>
#include <fstream>
#include <cstdint>

#include "crc32.hpp"

class Test {
public:
	Test(const std::string description, const uint32_t num);
	void static init(std::string fileName);
	void static end();
	void execute();
	void execute(std::function<void()> func);

	template<typename T>
	uint32_t CRC32(T* data, const size_t dataLength) {
		char* dataAsChar = (char*)data;
		return rc_crc32(0, dataAsChar, dataLength * sizeof(T));
	}

private:

	static std::ofstream reportStream;

	static std::chrono::time_point< std::chrono::high_resolution_clock > start;
	static std::chrono::time_point< std::chrono::high_resolution_clock > stop;

	std::string description;
	int numberOfIterations;

private:

	void report();
	void inline testStart();
	void inline testEnd();

protected:
	virtual void testSetup() = 0;
	virtual void testExecute() = 0;
	virtual void testDestroy() = 0;

	uint32_t verificationValue;
	uint32_t CRCValue;
};
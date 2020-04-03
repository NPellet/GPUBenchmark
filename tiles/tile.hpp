#pragma once

#include <stdlib.h>
#include <limits>
#include <mutex>
#include <map>
#include <assert.h>
#include <iostream>

class tile {
protected:

	size_t inline calcIndexAbs(const size_t x, const size_t y) const {
		return ((y - static_cast<size_t>(firstPointY)) * width + (x - static_cast<size_t>(firstPointX)));
	}

	size_t inline calcMaxIndex() const {
		return static_cast<size_t>(width) * static_cast<size_t>(height);
	}

public:
	// Self-tile properties
	const uint32_t width{ 0 };
	const uint32_t height{ 0 };
	unsigned int indexX{ 0 };
	unsigned int indexY{ 0 };
	int firstPointX{ 0 };
	int firstPointY{ 0 };
	int firstDataPointX{ 0 };
	int firstDataPointY{ 0 };

	tile() {}
	tile(uint32_t w, uint32_t h, int32_t posX, int32_t posY) : width(w), height(h), firstPointX(posX), firstPointY(posY) {}
	virtual ~tile() {}

	size_t inline calcIndex(const size_t x, const size_t y) const {
		return (y * width + x);
	}

	//virtual uint32_t getWidth() = 0;
	//virtual uint32_t getHeight() = 0;

	static void assertSameDimensions(tile* matA, tile* matB);
	static void assertSameDimensions(const tile* matA, const tile* matB);

	inline size_t getIndexAtAbsolute(const unsigned int x, const unsigned int y) const {
		return calcIndexAbs(x, y);
	}

	//	inline virtual void pointIntoStream(uint32_t x, uint32_t y, FILE* stream) = 0;
		//inline virtual void pointIntoBuffer(uint32_t x, uint32_t y, char* buffer, int& position) = 0;
};
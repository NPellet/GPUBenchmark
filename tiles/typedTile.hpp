#pragma once
#include "allocators/host.hpp"
#include "tile.hpp"

#include <limits>
#include <assert.h>
#include <string>
#include <iostream>

#if !defined(DEBUG_STDOUT)

#if defined(DEBUG) || defined(_DEBUG)
#define DEBUG_STDOUT( x ) std::cout << x << std::endl;
#else
#define DEBUG_STDOUT( x )
#endif

#endif

template<typename T, class Allocator = HostAllocator<T>>
class typedTile : public virtual tile, public virtual Allocator {
protected:

private:

	size_t memSize = 0;

public:

	//	using tile::getWidth;
	//	using tile::getHeight;

	T minValue = std::numeric_limits<T>::max();
	T maxValue = std::numeric_limits<T>::min();

	template<class OtherAllocator>
	typedTile(const typedTile<T, OtherAllocator>& other) : minValue(other.minValue), maxValue(other.maxValue) {	} // firstPointX(other.firstPointX), firstPointY(other.firstPointY), firstDataPointX(other.firstDataPointX), firstDataPointY(other.firstDataPointY)

	template<class OtherAllocator>
	typedTile(typedTile<T, OtherAllocator>&&) {
	}

	typedTile(uint32_t w, uint32_t h, uint32_t tileX, uint32_t tileY) : tile(w, h, tileX, tileY) {
		memSize = calcMemSize();
		allocate();
	}

	// Without tile positions
	typedTile(uint32_t w, uint32_t h) : typedTile<T, Allocator>(w, h, 0, 0) { }

	// Initalize from another matrix
	template<typename U>
	typedTile(typedTile<U>& from) : typedTile<T, Allocator>(from.getWidth(), from.getHeight(), from.firstPointX, from.firstPointY) {}

	~typedTile() {
	}

	size_t calcMemSize() override {
		return sizeof(T) * width * height;
	}

	// Size methods
	uint32_t getWidth() { return width; } // No override when not overriding HostALlocator
	uint32_t getHeight() { return height; }
	/*
	void reassign(typedTile<T>* other) {
		tile::assertSameDimensions(other, this);
		Allocator::free();
		hostData = other->getPtr();
		other->hostData = nullptr;

		if (!other->gpuIsExpired()) {
			std::shared_ptr<GPUMatrix<T>> newGPUMatrix = other->gpuAllocate(false, false);
			gpuPtr = std::weak_ptr<GPUMatrix<T>>(newGPUMatrix);
			//CacheManagement::placeInGPU(ptr); // Shared ptr (does not deallocate)
			newGPUMatrix.reset();
		}
	}*/

	void fillWith(T el) {
		for (size_t i = 0; i < calcMaxIndex(); ++i) {
			hostData[i] = el;
		}

		maxValue = el;
		minValue = el;
	}

	/**
	*		Feeds in a data in the absolute coordinate system
	*		That coordinate gets offseted by the starting point of the tilee
	*		Includes boundary check
	*		@todo move to assert ?
	*/
	void inline loadPointXYAbsolute(int32_t x, int32_t y, T value) {
		if (
			x - firstPointX >= 0 &&
			x < static_cast<int32_t>(width) + firstPointX &&
			y - firstPointY >= 0 &&
			y < static_cast<int32_t>(height) + firstPointY
			) {
			loadPoint((y - firstPointY) * width + (x - firstPointX), value);
		}
	}

	/**
	*		Feeds in a data in the relative coordinate system (where the tile starts at (0,0))
	*		Includes boundary check
	*		@todo move to assert ?
	*/
	void loadPointXY(int32_t x, int32_t y, T value) {
		if (
			x >= 0 &&
			x < width &&
			y >= 0 &&
			y < height
			) {
			loadPoint(calcIndex(x, y), value);
		}
	}

	void loadPointXY_assert(int32_t x, int32_t y, T value) {
		assert(x >= 0);
		assert(x < width);
		assert(y >= 0);
		assert(y < height);

		loadPoint(calcIndex(x, y), value);
	}

	void inline loadPoint(const size_t position, const T val) {
		assert(position < calcMaxIndex());
		hostData[position] = val;
	}

	inline T& getPoint(const size_t position) {
		return hostData[position];
	}

	inline T& getPoint(const uint32_t x, const uint32_t y) {
		return hostData[getIndexAtAbsolute(x, y)];
	}

	inline T& getPointXYAbsolute(const uint32_t x, const uint32_t y) {
		return getPoint(x, y);
	}

	inline T& getPointXY(const uint32_t x, const uint32_t y) {
		return getPoint(y * width + x);
	}

	void pointIntoStream(uint32_t x, uint32_t y, FILE* stream) {
		putc((int)hostData[calcIndexAbs(x, y)], stream);
	}

	void tileIntoStream(FILE* stream) {};
	void pointIntoBuffer(uint32_t x, uint32_t y, char* buffer, size_t& position) {
		memcpy(buffer + position, hostData + calcIndexAbs(x, y), sizeof(T));
		position += sizeof(T);
	}

	template<typename otherMatrix>
	void copyTo(typedTile<otherMatrix>* otherTile) {
		otherTile->allocateDataAndLock();

		for (uint32_t x = 0; x < width; x++) {
			for (uint32_t y = 0; y < height; y++) {
				otherTile->loadPointXY(x, y, (otherMatrix)this->getPointXY(x, y));
			}
		}
	}

	template<class OtherTile>
	void assertSameDimensions(OtherTile* matB) {
		assert(getWidth() == matB->getWidth());
		assert(getHeight() == matB->getHeight());
	}

	template<class OtherTile>
	void assertSameDimensions(const OtherTile* matB) {
		assert(getWidth() == matB->width);
		assert(getHeight() == matB->height);
	}
};

// PointIntoStream
//////////////////

// General definition (cannot be in class, refused by the MSVC compiler)
//template<typename T, class HostAllocator>
//void typedTile<T>::pointIntoStream(uint32_t x, uint32_t y, FILE* stream) {
//}
// Complete specialization for float
//template<> void typedTile<float2>::pointIntoStream(uint32_t x, uint32_t y, FILE* stream);
// Complete specialization for float
//template<> void typedTile<uint2>::pointIntoStream(uint32_t x, uint32_t y, FILE* stream);

//template<> void typedTile<unsigned char>::tileIntoStream(FILE* stream);
// PointIntoBuffer
//////////////////

// uint32 specialization
template<>
void typedTile<uint32_t>::pointIntoBuffer(uint32_t x, uint32_t y, char* buffer, size_t& position);
#pragma once

#include "tiledMatrix.hpp"
#include "typedTile.hpp"

#include <algorithm>
#include <vector>
#include <memory>
#include <set>

struct TileBoundaries {
	int16_t xFrom;
	int16_t xTo;
	int16_t yFrom;
	int16_t yTo;
};

struct TileList {
	using TileList1D = std::set<uint32_t>;
	TileList1D x;
	TileList1D y;
	void fill(TileList1D& list, uint32_t from, uint32_t to) {
		list.clear();
		for (; from < to; from++) {
			list.emplace(from);
		}
	}

	void fillX(uint32_t from, uint32_t to) {
		fill(this->x, from, to);
	}

	void fillY(uint32_t from, uint32_t to) {
		fill(this->y, from, to);
	}
};

template<typename T, class Tile = typedTile<T>>
class typedTiledMatrix : public tiledMatrix {
	using typedTile_t = Tile;
	using sharedTypedTile_t = std::shared_ptr<Tile>;

	struct tileReference {
		std::weak_ptr< typedTile_t > tilePtr;
		sharedTypedTile_t lockedTilePtr;
		bool locked;
		unsigned int firstPointX;
		unsigned int firstPointY;
		unsigned int firstDataPointX;
		unsigned int firstDataPointY;
		bool flag;

		unsigned int indexX;
		unsigned int indexY;
	};

private:
	typedTiledMatrix(const typedTiledMatrix&);
	typedTiledMatrix& operator=(const typedTiledMatrix&);

	std::vector< tileReference > tiles;

	void makeTiles(const unsigned int tileWidth, const unsigned int tileHeight, const unsigned int overlap) {
		unsigned int i = 0, j = 0;// , counter = 0;

		this->tileWidth = tileWidth;
		this->tileHeight = tileHeight;
		this->overlap = overlap;

		tileNumX = (unsigned int)((width + tileWidth - 1) / tileWidth);
		tileNumY = (unsigned int)((height + tileHeight - 1) / tileHeight);
		numberOfTiles = tileNumX * tileNumY;
		tiles.reserve(tileNumX * tileNumY);

		for (j = 0; j < tileNumY; j++) {
			for (i = 0; i < tileNumX; i++) {
				tileReference ref;
				//				ref.tilePtr = nullptr;
				ref.lockedTilePtr = nullptr;
				ref.firstPointX = tileWidth * i - overlap;
				ref.firstPointY = tileHeight * j - overlap;
				ref.firstDataPointX = tileWidth * i;
				ref.firstDataPointY = tileHeight * j;
				ref.flag = false;
				ref.locked = false;

				ref.indexX = i;
				ref.indexY = j;

				tiles.push_back(ref);
			}
		}
	}

public:

	T minValue = std::numeric_limits<T>::max();
	T maxValue = std::numeric_limits<T>::min();

	template<typename U, class UAllocator>
	typedTiledMatrix(typedTiledMatrix<U, UAllocator>* tileToCopyStructureFrom) {
		width = tileToCopyStructureFrom->getWidth();
		height = tileToCopyStructureFrom->getHeight();
		tileWidth = tileToCopyStructureFrom->getTileWidth();
		tileHeight = tileToCopyStructureFrom->getTileHeight();
		overlap = tileToCopyStructureFrom->getTileOverlap();
		makeTiles(tileWidth, tileHeight, overlap);
	}
	/*
	typedTiledMatrix(const unsigned int w, const unsigned int h, const unsigned int o, unsigned int tileWidth, unsigned int tileHeight, cudaStream_t* stream) : typedTiledMatrix(w, h, o, tileWidth, tileHeight) {
		cudaStream = stream;
	}*/

	typedTiledMatrix(const unsigned int w, const unsigned int h, const unsigned int o, unsigned int tileWidth, unsigned int tileHeight) {
		width = w;
		height = h;

		if (tileWidth > width || tileWidth == 0) {
			tileWidth = width;
		}

		if (tileHeight > height || tileHeight == 0) {
			tileHeight = height;
		}

		makeTiles(tileWidth, tileHeight, o);
	}

	typedTiledMatrix(const unsigned int w, const unsigned int h, const unsigned int o) : typedTiledMatrix(w, h, o, 0, 0) {}

	~typedTiledMatrix() {
		freeTiles();
	}

	size_t tileMemorySize() {
		return sizeof(T) * (tileWidth + overlap * 2) * (tileHeight + overlap * 2);
	}

	inline int calcTileIndex(const int x, const int y) {
		unsigned int index = y * tileNumX + x;

		//	assert(index >= 0);
		assert(index < numberOfTiles);
		return index;
	}
	/**
	*	Returns the locked ptr is it exists.
	*	If not, tries the weak pointer (let's say the tile is cached). If successful, returns it.
	*	If not, throws an exception
	*/
	inline sharedTypedTile_t getTile(const unsigned int i, const bool lock = false) {
		//		assert(i >= 0);
		assert(i < numberOfTiles);

		tileReference& ref = tiles[i];

		if (ref.locked) {
			return ref.lockedTilePtr;
		}

		if (ref.tilePtr.expired()) {
			throw "Tile does not exist anymore";
		}

		if (lock) {
			ref.locked = true;
			ref.lockedTilePtr = ref.tilePtr.lock();
			return ref.lockedTilePtr;
		}

		return ref.tilePtr.lock();
	}

	inline sharedTypedTile_t getTileXY(const unsigned int x, const unsigned int y) {
		return getTile(calcTileIndex(x, y));
	}

	sharedTypedTile_t reassign(std::shared_ptr<typedTile<T>> otherTile, const unsigned int x, const unsigned int y) {
		sharedTypedTile_t thisTile = allocateTileFromXY(x, y);
		tile::assertSameDimensions(thisTile.get(), otherTile.get());
		thisTile->reassign(otherTile.get());
		return thisTile;
	}

	inline void freeTile(const unsigned int i) {
		//		assert(i >= 0);
		assert(i < numberOfTiles);

		tileReference& ref = tiles[i];

		if (ref.locked) {
			ref.lockedTilePtr.reset();
			ref.locked = false;
		}
	}

	void freeTiles() {
		for (unsigned int i = 0; i < tileNumX * tileNumY; i++) {
			freeTile(i);
		}
	}

	/**
	*	Returns the locked ptr is it exists.
	*	If not, tries the weak pointer (let's say the tile is cached). If successful, returns it.
	*	If not, allocates the tile and returns a shared_ptr to it.
	*/
	sharedTypedTile_t allocateTile(const unsigned int i, const bool lock = false, const T defaultValue = 0) {
		//		assert(i >= 0);
		assert(i < numberOfTiles);

		tileReference& ref = tiles[i];

		if (ref.locked) {
			return ref.lockedTilePtr;
		}

		if (ref.tilePtr.expired()) {
			auto sharedPtr = std::make_shared< typedTile_t >(tileWidth + overlap * 2, tileHeight + overlap * 2, ref.firstPointX, ref.firstPointY);
			sharedPtr->fillWith(defaultValue);
			//sharedPtr->cudaSetStream(cudaStream);
			sharedPtr->indexX = ref.indexX;
			sharedPtr->indexY = ref.indexY;

			ref.tilePtr = sharedPtr; // Decay to weak pointer

			if (lock == true) {
				ref.locked = true;
				ref.lockedTilePtr = sharedPtr;
			}

			return sharedPtr;
		}

		if (lock == true) {
			ref.locked = true;
			ref.lockedTilePtr = ref.tilePtr.lock();
		}

		return ref.tilePtr.lock();
	}

	inline sharedTypedTile_t allocateTileFromXY(const unsigned int x, const unsigned int y, bool lock = false) {
		return allocateTile(calcTileIndex(x, y), lock);
	}

	inline bool isTileAllocated(const int i) {
		return !tiles[i].tilePtr.expired();
	}

	inline bool isTileAllocatedXY(const unsigned int x, const unsigned int y) {
		return isTileAllocated(calcTileIndex(x, y));
	}

	/** Tile flagging */
	inline void flagTile(const int i, bool flag) {
		assert(i >= 0);
		assert(i < numberOfTiles);

		tiles[i].flag = flag;
	}

	void flagTileXY(const int x, const int y, const bool flag) {
		return flagTile(calcTileIndex(x, y), flag);
	}

	void flagTiles(const bool flag) {
		for (int i = 0; i < numberOfTiles; i++) {
			flagTile(i, flag);
		}
	}

	inline bool isTileFlagged(const int i) {
		assert(i >= 0);
		assert(i < numberOfTiles);

		return tiles[i].flag;
	}

	inline bool isTileFlaggedXY(const int x, const int y) {
		return isTileFlagged(calcTileIndex(x, y));
	}

	/**
	*	Returns the locked ptr is it exists.
	*	If not, tries the weak pointer (let's say the tile is cached). If successful, returns it.
	*	If not, allocates the tile and returns a shared_ptr to it.
	*/
	inline sharedTypedTile_t lockTile(const int i) {
		return getTile(i, true);
	}

	inline void unlockTile(const int i) {
		assert(i >= 0);
		assert(i < numberOfTiles);

		tileReference& ref = tiles[i];

		if (ref.locked) {
			ref.locked = false;
			ref.lockedTilePtr.reset();
		}
	}

	inline bool isTileLocked(const int i) {
		assert(i >= 0);
		assert(i < numberOfTiles);

		return tiles[i].locked;
	}

	inline bool isTileLockedXY(const int x, const int y) {
		return tiles[calcTileIndex(x, y)].locked;
	}

	void lockTiles() {
		for (uint32_t i = 0; i < numberOfTiles; i++) {
			allocateTile(i, true);
		}
	}

	void unlockTiles() {
		for (int i = 0; i < tileNumX * tileNumY; i++) {
			//	tiles[i]->HostDeallocate();
			unlockTile(i);
		}
	}

	inline void getTileBoundaries(const int x, const int y, int& xFrom, int& yFrom, int& xTo, int& yTo) {
		xFrom = tileWidth * x - overlap;
		yFrom = tileHeight * y - overlap;
		xTo = tileWidth * (x + 1) + overlap;
		yTo = tileHeight * (y + 1) + overlap;
	}

	void retrieveAllTilesFromGPU() {
		for (int i = 0; i < tileNumX * tileNumY; i++) {
			getTile(i)->gpuRetrieve();
		}
	}

	void loadLineFromTiles(T* buffer, uint32_t y) {
		int tileX, tileY;
		tileY = y / tileHeight; // Automatic flooring
		for (uint32_t x = 0; x < width; x++) {
			tileX = x / tileWidth; // Automatic flooring
			buffer[x] = getTileXY(tileX, tileY)->getPoint(x, y);// Need to cast the matrix element into the tiff element
		}
	}

	void loadLineFromTiles(char* buffer, uint32_t y) {
		int tileX, tileY;
		tileY = y / tileHeight; // Automatic flooring
		size_t index;

		for (uint32_t x = 0; x < width; x++) {
			tileX = x / tileWidth; // Automatic flooring

			T* src = getTileXY(tileX, tileY)->getHostDataPtr();
			index = getTileXY(tileX, tileY)->getIndexAtAbsolute(x, y);

			memcpy(buffer + x * sizeof(T), src[index], sizeof(T));
		}
	}

	void offsetOrigin(uint32_t dx, uint32_t dy) {
		for (int i = 0; i < tileNumX * tileNumY; i++) {
			if (isTileAllocated(i)) {
				sharedTypedTile_t tile = getTile(i);
				tile->firstPointX += dx;
				tile->firstPointY += dy;
				tile->firstDataPointX += dx;
				tile->firstDataPointX += dy;
			}

			tiles[i].firstPointX += dx;
			tiles[i].firstPointY += dy;
			tiles[i].firstDataPointX += dx;
			tiles[i].firstDataPointX += dy;
		}
	}

	TileBoundaries findBoundaries(uint32_t xFrom, uint32_t xTo, uint32_t yFrom, uint32_t yTo) {
		TileBoundaries boundaries;

		uint32_t tileCol = xTo / tileWidth;
		uint32_t tileRow = yTo / tileHeight;

		boundaries.xTo = tileCol;
		if (tileWidth - xTo % tileWidth < overlap && tileCol < tileNumX - 1) {
			boundaries.xTo += 1;
		}

		boundaries.yTo = tileRow;
		if (tileHeight - yTo % tileHeight < overlap && tileRow < tileNumY - 1) {
			boundaries.yTo += 1;
		}

		tileCol = xFrom / tileWidth;
		tileRow = yFrom / tileHeight;

		boundaries.xFrom = tileCol;
		if (xFrom % tileWidth < overlap && tileCol > 0) {
			boundaries.xFrom -= 1;
		}

		boundaries.yFrom = tileRow;
		if (yFrom % tileHeight < overlap && tileRow > 0) {
			boundaries.yFrom -= 1;
		}

		return boundaries;
	}

	TileList findBoundariesList(uint32_t xFrom, uint32_t xTo, uint32_t yFrom, uint32_t yTo) {
		TileBoundaries b = findBoundaries(xFrom, xTo, yFrom, yTo);

		TileList list;
		list.fillX(b.xFrom, b.xTo);
		list.fillY(b.yFrom, b.yTo);
		return list;
	}

	void loadLineIntoTiles(T* buffer, uint32_t y) {
		uint32_t tileRow = y / tileHeight, tileCol;
		bool nextY = false;
		bool nextX = false;
		bool prevX = false;
		bool prevY = false;
		typedTile_t* cachedTile = nullptr;

		for (int i = 0; i < tileNumX * tileNumY; i++) {
			allocateTile(i, true, 0);
		}

		if (tileHeight - y % tileHeight < overlap && tileRow < tileNumY - 1) {
			nextY = true;
		}

		if (y % tileHeight < overlap && tileRow > 0) {
			prevY = true;
		}

		uint32_t prevTileCol = 0;
		//	for (uint32 i = 0; i < numberOfTiles; i++) {
		for (uint32_t x = 0; x < width; x++) {
			tileCol = x / (tileWidth);

			if (overlap == 0) {
				if (tileCol != prevTileCol || cachedTile == nullptr) {
					cachedTile = getTileXY(tileCol, tileRow).get();
				}

				cachedTile->loadPointXYAbsolute(x, y, buffer[x]);
				prevTileCol = tileCol;
				continue;
			}

			if (tileWidth - x % tileWidth < overlap && tileCol < tileNumX - 1) {
				nextX = true;
			}
			else {
				nextX = false;
			}

			if (x % tileWidth < overlap && tileCol > 0) {
				prevX = true;
			}
			else {
				prevX = false;
			}

			getTileXY(tileCol, tileRow)->loadPointXYAbsolute(x, y, buffer[x]);
			if (nextY) {
				getTileXY(tileCol, tileRow + 1)->loadPointXYAbsolute(x, y, buffer[x]);

				if (nextX) {
					getTileXY(tileCol + 1, tileRow + 1)->loadPointXYAbsolute(x, y, buffer[x]);
				}
				if (prevX) {
					getTileXY(tileCol - 1, tileRow + 1)->loadPointXYAbsolute(x, y, buffer[x]);
				}
			}

			if (nextX) {
				getTileXY(tileCol + 1, tileRow)->loadPointXYAbsolute(x, y, buffer[x]);
			}

			if (prevY) {
				getTileXY(tileCol, tileRow - 1)->loadPointXYAbsolute(x, y, buffer[x]);

				if (prevX) {
					getTileXY(tileCol - 1, tileRow - 1)->loadPointXYAbsolute(x, y, buffer[x]);
				}

				if (nextX) {
					getTileXY(tileCol + 1, tileRow - 1)->loadPointXYAbsolute(x, y, buffer[x]);
				}
			}

			if (prevX) {
				getTileXY(tileCol - 1, tileRow)->loadPointXYAbsolute(x, y, buffer[x]);
			}
		}
	}

	void fillWith(T el) {
		for (uint32_t i = 0; i < numberOfTiles; i++) {
			getTile(i)->fillWith(el);
		}
	}

	void minMaxValueFromTiles() {
		if (numberOfTiles == 0) { // Only if the tile has no data by itself but has only sub-tiles
			return;
		}

		minValue = getTile(0)->minValue;
		maxValue = getTile(0)->maxValue;

		for (uint32_t i = 1; i < numberOfTiles; i++) {
			minValue = std::min<T>(minValue, getTile(i)->minValue);
			maxValue = std::max<T>(maxValue, getTile(i)->maxValue);
		}
	}

	inline void pointIntoStream(uint32_t x, uint32_t y, FILE* stream) {
		putc(getTileFromPosition(x, y)->getPointXYAbsolute(x, y), stream);
	}
};

void typedTiledMatrix<uint32_t, typedTile<uint32_t>>::loadLineFromTiles(char* buffer, uint32_t y);
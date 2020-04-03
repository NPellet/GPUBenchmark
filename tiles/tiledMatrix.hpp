#pragma once

#include "tile.hpp"

#include <stdint.h>
#include <stdlib.h>
#include <limits>
#include <assert.h>

class tiledMatrix {
protected:
	tile** tiles;

public:

	uint32_t numberOfTiles = 0;
	uint32_t tileNumX = 0;
	uint32_t tileNumY = 0;
	uint32_t tileWidth = 0;
	uint32_t tileHeight = 0;
	uint32_t overlap;

	uint32_t width = 0;
	uint32_t height = 0;

	inline uint32_t getWidth() { return width; }
	inline uint32_t getHeight() { return height; }

	// Child tiles properties

	inline uint32_t getTileWidth() {
		return tileWidth;
	}

	inline uint32_t getTileHeight() {
		return tileHeight;
	}

	inline uint32_t getTileOverlap() {
		return overlap;
	}
	//tile* getTileFromPosition(int32_t x, int32_t y);
	void checkSameStructureAs(tiledMatrix* otherTile);
	virtual void minMaxValueFromTiles() = 0;
};
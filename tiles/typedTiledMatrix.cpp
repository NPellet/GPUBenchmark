#include "typedTiledMatrix.hpp"

template<>
void typedTiledMatrix<uint32_t>::loadLineFromTiles(char* buffer, uint32_t y) {
	int32_t tileX, tileY;
	tileY = y / tileHeight; // Automatic flooring
//	size_t index;

	for (uint32_t x = 0; x < width; x++) {
		tileX = x / tileWidth; // Automatic flooring

		uint32_t point = getTileXY(tileX, tileY)->getPoint(x, y);

		buffer[x * 3] = point >> 24;
		buffer[x * 3 + 1] = point >> 16;
		buffer[x * 3 + 2] = point >> 8;

		//memcpy( buffer + x * 3, src + index, 3 );// Need to cast the matrix element into the tiff element
	}
}
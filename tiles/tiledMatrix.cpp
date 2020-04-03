#include "tiledMatrix.hpp"

void tiledMatrix::checkSameStructureAs(tiledMatrix* otherTile) {
	if (
		tileWidth != otherTile->tileWidth ||
		tileHeight != otherTile->tileHeight ||
		overlap != otherTile->overlap ||
		width != otherTile->width ||
		height != otherTile->height
		) {
#ifdef ERROR_MATRICES_STRUCT_MATCH
		throw ERROR_MATRICES_STRUCT_MATCH;
#else
		std::cerr << "Matrix structures don't match" << std::endl;
		exit(EXIT_FAILURE);
#endif
	}
}
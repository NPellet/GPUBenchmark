#pragma once
#include "tiffio.h"

template<class Tile>
Tile* loadTiff(const char* tiffFile) {
	TIFF* tif = TIFFOpen(tiffFile, "r");

	if (tif) {
		uint32 w, h;

		uint16 bitPerSample;
		uint16 samplePerPx;
		uint16 planar;

		uint16 photo;

		uint16 sampleformat;

		TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
		TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);

		TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &bitPerSample);
		TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplePerPx);

		TIFFGetField(tif, TIFFTAG_PLANARCONFIG, &planar);
		TIFFGetField(tif, TIFFTAG_PHOTOMETRIC, &photo);
		TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sampleformat);

		Tile* mat = new Tile(w, h);
		for (unsigned int i = 0; i < h; i++) {
			TIFFReadScanline(tif, &(mat->getPtr()[i * w]), i, 0);
		}

		TIFFClose(tif);
		return mat;
	}

	TIFFClose(tif);
	std::cerr << "Tiff file could not opened" << std::endl;
};
#pragma once
#include <stdint.h>
#include <iostream>
#include "tiffio.h"
#include "../../test/test.hpp"
#include "../../tiles/typedTile.hpp"
#include "../../tiles/allocators/gpumatrix.cuh"
//#include "../../../relief_calculator/tiff.cuh"

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

namespace Tests {
	namespace Distance {
		class Base : public Test {
		public:

			using Tile = typedTile<unsigned char, GPUMatrixAllocator<unsigned char>>;
			using TileDistance = typedTile<unsigned int, GPUMatrixAllocator<unsigned int>>;

			Base(const std::string filePath, const std::string descr, uint32_t numRep) : Test(descr, numRep) {
				source = loadTiff<Tile>(filePath.c_str());
				distance = new TileDistance{ source->width, source->height };
			}

			~Base() {
			}

		protected:
			Tile* source;
			TileDistance* distance;
			typedTile<uint2, GPUMatrixAllocator<uint2>>* distance_2comp;

			// Tiles are allocated on the stack, so no need to do memory management in the test setup and destroy
			void testSetup() override {
			};

			void testDestroy() override {
				distance->gpuRetrieve();
				CRCValue = CRC32(distance->getPtr(), static_cast<size_t>(distance->width) * distance->height);

				delete source;
				delete distance;
			};
		};
	}
}
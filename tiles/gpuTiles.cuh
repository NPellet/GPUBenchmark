#pragma once

#include "typedTile.hpp"
#include "typedTiledMatrix.hpp"
#include "allocators/gpumatrix.cuh"
#include "allocators/gpumatrixlru.cuh"

template<typename T, class Allocator>
class _GPUTile : public virtual Allocator, public virtual typedTile<T, HostPinnedAllocator<T>> {
public:

	_GPUTile(uint32_t w, uint32_t h, uint32_t tileX, uint32_t tileY) : typedTile<T, Allocator>(w, h, tileX, tileY) {
	}

	// Without tile positions
	_GPUTile(uint32_t w, uint32_t h) : _GPUTile(w, h, 0, 0) { }

	uint32_t getWidth() {
		return width;
	}
	uint32_t getHeight() {
		return height;
	}
};

template<typename T>
using GPUTile = _GPUTile<T, GPUMatrixAllocator<T>>;

template<typename T, class Allocator>
class _GPULRUTile : public virtual Allocator, public virtual GPUTile<T> {
};

template<typename T>
using GPULRUTile = _GPULRUTile<T, GPULRUAllocator<T>>;

template<typename T>
using GPUMatrix = typedTiledMatrix<T, GPUTile<T>>;

template<typename T>
using GPULRUMatrix = typedTiledMatrix<T, GPULRUTile<T>>;
#pragma once

#include "hostpinned.hpp"
#include "../gpumatrix.cuh"

#include <cuda_runtime.h>
#include <memory>

template<typename T>
class GPUMatrixAllocator : public virtual HostPinnedAllocator<T> {
private:
	uint32_t roundUpToMult(uint32_t val) {
		return pow(2, ceil(log(val) / log(2)));
	}
public:
	GPUMatrixAllocator() { }

	~GPUMatrixAllocator() {
		this->free();
	}
	dim3 dimBlock;
	dim3 dimGrid;
	// Needed for the calculation of dimBlock and dimGrid
	virtual uint32_t getWidth() = 0;
	virtual uint32_t getHeight() = 0;

	void allocate() override {
		constexpr int numberThreadPerBlock = 32;

		dimBlock = { min(numberThreadPerBlock, roundUpToMult(getWidth())), min(numberThreadPerBlock, roundUpToMult(getHeight())) }; //Make the block an even number

		if (getWidth() == 1) {
			dimBlock.x = 1;
		}

		if (getHeight() == 1) {
			dimBlock.y = 1;
		}

		dimGrid = { (getWidth() + dimBlock.x - 1) / dimBlock.x, (getHeight() + dimBlock.y - 1) / dimBlock.y };

		HostPinnedAllocator<T>::allocate();
	}

	dim3 calcDimGrid(unsigned int threadSize) {
		return dim3((getWidth() + threadSize - 1) / threadSize, (getHeight() + threadSize - 1) / threadSize, 1);
	}

	dim3 calcDimGrid(const dim3& dimBlock) {
		return dim3(
		(getWidth() + dimBlock.x - 1) / dimBlock.x,
			(getHeight() + dimBlock.y - 1) / dimBlock.y,
			1
			);
	}

	dim3 calcDimBlock(unsigned int threadSize) {
		return dim3(threadSize, threadSize, 1);
	}

	void cudaSetStream(cudaStream_t* stream) {
		cudaStream = stream;
	}

	cudaStream_t cudaGetStream() {
		if (cudaStream == nullptr) {
			return (cudaStream_t)0;
		}
		else {
			return *cudaStream;
		}
	}

	virtual inline T* gpuGetPtr() {
		return gpuGetMatrix()->getDataPtr();
	}

	std::shared_ptr<GPUMatrixImpl<T>> gpuGetMatrix() {
		return gpuPtr;
	}

	virtual std::shared_ptr<GPUMatrixImpl<T>> inline gpuAllocate(bool copy = true, bool forceCopy = false) {
		if (gpuPtr != nullptr) {
			if (forceCopy) {
				gpuPtr->copyFrom();
			}

			return gpuPtr; // Already allocated
		}

		gpuPtr = gpuCreateMatrix(copy);
		return gpuPtr;
	}

	virtual void gpuDeallocate(bool copy = true) {
		if (gpuIsExpired()) {
			return;
		}
		// Should we copy back to host before deallocation ?
		if (copy) {
			gpuGetMatrix()->copyTo();
		}

		gpuFree();
	}

	void gpuClear() { // Alias to GPUDeallocate without copying
		gpuDeallocate(false);
	}

	void gpuRetrieve() {
		if (gpuIsExpired()) {
			return;
		}
		gpuGetMatrix()->copyTo();
	}

	std::shared_ptr<GPUMatrixImpl<T>> gpuAssertAllocation() {
		assert(!gpuIsExpired());
		return gpuGetMatrix();
	}

private:

	std::shared_ptr<GPUMatrixImpl<T>> gpuPtr;
	cudaStream_t* cudaStream = nullptr;

protected:

	// Host free
	void free() override {
		// Free host data
		HostPinnedAllocator<T>::free();
		if (gpuPtr != nullptr) { // Invalidate the host with regards to the GPU
			gpuPtr->invalidateHost();
		}

		// Free GPU data
		gpuFree();
	}

	// GPU free automatically handles by GPUMatrix<T>
	virtual void gpuFree() {
		gpuPtr.reset();
	}

	// Determined whether the GPU memory has been freed already
	virtual bool gpuIsExpired() {
		return gpuPtr == nullptr;
	}

	// Creates a new matrix and returns a shared pointer to it
	std::shared_ptr<GPUMatrixImpl<T>> inline gpuCreateMatrix(const bool copy) {
		auto matrix = std::make_shared<GPUMatrixImpl<T>>(hostData, calcMemSize()); // Shared ptr
	//	matrix->cudaSetStream(cudaStream);

		if (copy) {
			matrix->copyFrom();
		}
		return matrix;
	}
};
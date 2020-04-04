#pragma once

#include "gpumatrix.cuh"

template<typename T>
class GPULRUAllocator : public virtual GPUMatrixAllocator<T> {
private:

	std::weak_ptr<GPUMatrixImpl<T>> gpuPtr; // We only store a weak pointer, which will deallocate when the cache looses it

public:
	// Needed for the calculation of dimBlock and dimGrid

	virtual void gpuFree() override {
		if (gpuIsExpired()) { // No hard kill if the memory has already been freed
			return;
		}
		CacheManagement::removeAllGPUReferencesTo(gpuGetMatrix());
	}

	virtual bool gpuIsExpired() override {
		return gpuPtr.expired();
	}

	std::shared_ptr<GPUMatrixImpl<T>> gpuGetMatrix() {
		if (gpuIsExpired()) {
			std::cerr << "GPU data is expired. Cannot access it" << std::endl;
			exit(EXIT_FAILURE);
		}

		return gpuPtr.lock();
	}

	std::shared_ptr<GPUMatrixImpl<T>> gpuAllocate(bool copy = true, bool forceCopy = false) {
		if (!gpuIsExpired()) {
			if (forceCopy) { // Exist but we want to force-copy the host data to the device memory
				gpuPtr.lock()->copyFrom();
			}
			CacheManagement::placeInGPU(gpuPtr.lock());
			return gpuPtr.lock();
		}

		// We are going to allocate new memory. We want to free some first if needed
		CacheManagement::reserveGPULRU(calcMemSize());

		auto matrix = this->GPUMatrixAllocator<T>::gpuCreateMatrix(copy);
		CacheManagement::placeInGPU(matrix); // Shared ptr (does not deallocate)

		gpuPtr = matrix; // Decay to local weak pointer to allow deallocation in case of cache cycling

		if (copy) {
			matrix->copyFrom();
		}

		return matrix;
	}

	void gpuClear() { // Alias to GPUDeallocate
		gpuDeallocate(false);
	}

	std::shared_ptr<GPUMatrixImpl<T>>->gpuAssertAllocation() {
		assert(!gpuIsExpired());

		return gpuPtr.lock();
	}
};
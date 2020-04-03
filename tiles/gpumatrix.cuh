#pragma once

#include "cuda_runtime.h"
#include <iostream>
#include <cstring>

class DataMatrix {
protected:
	size_t memSize;

public:
	size_t calcMemSize() {
		return this->memSize;
	}
};

template<typename T>
class TypedDataMatrix : public DataMatrix {
protected:
	T* data = nullptr;

public:
};

template<typename T>
class GPUMatrixImpl : public TypedDataMatrix<T> {
private:

	//HostMatrix<T>* mate;
	T* matePtr;
	cudaStream_t* cudaStream = nullptr;

public:
	GPUMatrixImpl(T* hostPtr, size_t memSize) {
		checkCuda(cudaMalloc((void**)&(this->data), memSize));

		this->memSize = memSize;
		matePtr = hostPtr;
	}

	~GPUMatrixImpl() {
		if (this->data == nullptr) {
			return;
		}

		if (matePtr != nullptr) {
			copyTo();
		}

		// Before freeing up the data, we need to synchronize the stream, which, up until now, was completely asynchronous with regards to the host
		// I think this is already covered by the final device -> host data transfer that's anyways internally called before memory releasing
		//cudaStreamSynchronize(*cudaStream);

		// Now that all kernels + data transfers are done, we can finally free up the memory
		checkCuda(cudaFree(this->data));
		this->data = nullptr;
	}

	void invalidateHost() {
		matePtr = nullptr;
	}

	void cudaSetStream(cudaStream_t* stream) {
		cudaStream = stream;
	}

	void copyFrom() {
		if (cudaStream != nullptr) {
			checkCuda(cudaMemcpyAsync(this->data, matePtr, this->memSize, cudaMemcpyHostToDevice, *cudaStream));
		}
		else {
			checkCuda(cudaMemcpy(this->data, matePtr, this->memSize, cudaMemcpyHostToDevice));
		}
	}

	void copyTo() {
		if (cudaStream != nullptr) {
			checkCuda(cudaMemcpyAsync(matePtr, this->data, this->memSize, cudaMemcpyDeviceToHost, *cudaStream));

			// Here are trying to retrieve the data to the host.
			// We therefore need to block the host until all operations (kernels + data transfers) have completed.
			// Only then can we ensure that the host data is valid
			cudaStreamSynchronize(*cudaStream);
		}
		else {
			// AFAIK, in the default stream, cudaMemcpy is host-blocking and automatically synchronises the default stream with the host
			checkCuda(cudaMemcpy(matePtr, this->data, this->memSize, cudaMemcpyDeviceToHost));
		}
	}

	inline T* getDataPtr() {
		return this->data;
	}

	inline T* getPtr() {
		return this->data;
	}
};
#pragma once

#include "host.hpp"
#include <string>
#include <cuda_runtime.h>
#include <cassert>

template<typename T>
class HostPinnedAllocator : public virtual HostAllocator<T> {
public:
	HostPinnedAllocator() {
	}

	~HostPinnedAllocator() {
	}

protected:

	void allocate() {
#ifdef DEBUG_STDOUT
		DEBUG_STDOUT("Allocating " << std::to_string(calcMemSize()) << " bytes to Host");
#endif

		checkCuda(cudaMallocHost((void**)&(hostData), calcMemSize()));
		memset(hostData, 0, calcMemSize());
	}

	void free() {
		if (hostData == nullptr) {
			return;
		}

#ifdef DEBUG_STDOUT
		//	DEBUG_STDOUT("De-Allocating " << std::to_string(calcMemSize()) << " bytes to Host");
#endif

		checkCuda(cudaFreeHost(hostData));
		hostData = nullptr; // Reset the pointer
	}
};
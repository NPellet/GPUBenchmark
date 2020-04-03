#pragma once
#include <iostream>

template<typename T>
class HostAllocator {
private:

protected:
	T* hostData = nullptr;

	virtual void allocate() {};
	virtual void free() {};

public:
	HostAllocator() {};

	HostAllocator(const HostAllocator& other) {
		hostData = other.hostData;
	}

	~HostAllocator() {
	}

	T* getPtr() {
		return hostData;
	}

	virtual size_t calcMemSize() = 0;
};
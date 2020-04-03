#pragma once
// Taken from https://rosettacode.org/wiki/CRC-32

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

uint32_t rc_crc32(uint32_t crc, const char* buf, size_t len);

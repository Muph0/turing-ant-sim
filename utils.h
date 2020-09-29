#pragma once

#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#define mod(A,B) ((((A)%(B))+(B))%(B))
#define max(A,B) ((A)>(B)?(A):(B))
#define min(A,B) ((A)<(B)?(A):(B))
#define sign(A) ((A) == 0 ? 0 : ((A) > 0 ? 1 : -1))
#define clamp(X,A,B) max((A),min((B),(X)))
//#define rnd_next(state) (state = (state >> 1) | (((state >> 0) ^ (state >> 10) ^ (state >> 30) ^ (state >> 31)) << 31))
#define byte_positive(B) ((B)>0&&(B)<128)
#define byte_negative(B) ((B)<0&&(B)>=128)
#define uppercase(c) ((c)>='a'?(c)-'a'+'A':(c))

// to indicate that no break is intentional
#define no_break

#define get_bits(number, lo, count) ( (number) >> (lo) & ~( ~0ui64 << (count) ) )
#define set_bits(number, lo, count, value) ( (number) = (number) & ~(~( ~0ui64 << (count) ) << (lo)) | (~( ~0ui64 << (count) ) & (value) ) << (lo) )
#define get_bit(number, bit) get_bits(number, bit, 1)
#define set_bit(number, bit, value) set_bits(number, bit, 1, value)

#define try_set_symbol(FIELD) if (token == #FIELD ":") FIELD = value

#define BYTE_TO_BINARY_PATTERN "%c%c%c%c%c%c%c%c"
#define BYTE_TO_BINARY(BYTE)  \
  ((BYTE) & 0x80 ? '1' : '0'), \
  ((BYTE) & 0x40 ? '1' : '0'), \
  ((BYTE) & 0x20 ? '1' : '0'), \
  ((BYTE) & 0x10 ? '1' : '0'), \
  ((BYTE) & 0x08 ? '1' : '0'), \
  ((BYTE) & 0x04 ? '1' : '0'), \
  ((BYTE) & 0x02 ? '1' : '0'), \
  ((BYTE) & 0x01 ? '1' : '0') 

#define nameof(x) #x

#if defined(__CUDACC__)
#define __kernel_call(BLOCKS, THREADS) <<< BLOCKS, THREADS >>>
#else
#define __kernel_call(BLOCKS, THREADS)
#endif

#define __both__ __host__ __device__

#if !defined(__CUDACC__)
#define __syncthreads() 
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) while (1) {}
	}
}

__device__ inline void lfsr32(uint32_t* state)
{
	//32,22,2,1
	int bit = (*state >> 0) ^ (*state >> 10) ^ (*state >> 30) ^ (*state >> 31);
	*state = (*state >> 1) | (bit << 31);
}
__device__ uint32_t rnd_next(uint32_t& state);
__device__ int int_amount(float amount, uint32_t& rand_state);

int from_binary(const char* str);
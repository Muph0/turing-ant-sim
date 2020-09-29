#include "utils.h"

__device__ uint32_t rnd_next(uint32_t& state)
{
	uint32_t result = state;
	for (int i = 0; i < 8 * sizeof(state); i++)
	{
		lfsr32(&state);
	}

	return result;
}

#include <exception>

__device__ int int_amount(float amount, uint32_t& rand_state)
{
	if (abs(amount) >= 1)
		return (int)amount;

	int dice = (int)(1 / abs(amount));
	return (rnd_next(rand_state) % dice == 0) * sign(amount);
}

int from_binary(const char* str)
{
	int result = 0;
	for (int i = 0; i < 32; i++)
	{
		if (str[i] == 0) break;
		result <<= 1;
		if (str[i] == '0') continue;
		if (str[i] == '1') result |= 1;
		else throw std::exception("Bad binary string.");
	}

	return result;
}
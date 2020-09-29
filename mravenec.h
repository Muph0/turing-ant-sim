#pragma once

#include <cstdint>
#include <fstream>
#include <cuda_runtime.h>

#include "utils.h"

constexpr int UNIT_MEM_SIZE = 256;
constexpr size_t UNIT_PROGMEM_SIZE = 128;
constexpr size_t UNIT_DYNMEM_SIZE = UNIT_MEM_SIZE - UNIT_PROGMEM_SIZE;
constexpr size_t UNIT_PROGMEM_IRCOUNT = UNIT_PROGMEM_SIZE * 2;

struct Mravenec;

#include "level.h"

enum class IR
{
	Next = 0, Prev, Increment, Decrement, WhileDo, Done, Add, Sub, ShiftL, ShiftR, Move, Sense, Bite, LayEgg, Measure, Burn
};
enum class Direction
{
	Right = 0, Down, Left, Up,
	PosX = 0, PosY, NegX, NegY
};
__constant__ const char dx[4] = { 1, 0, -1, 0 };
__constant__ const char dy[4] = { 0, 1, 0, -1 };

#define dir_left(d) ((d) + 3 & 3)
#define dir_right(d) ((d) + 1 & 3)
#define dir_back(d) ((d) + 2 & 3)

constexpr char DIRECTION_CHAR[] = {
	'>', 'v', '<', '^', 
	26, 25, 27, 24,		  //  outline arrows
	17, 31, 16, 30,       //  full arrows
};
constexpr char IR_MNEM[] = { '>', '<', '+', '-', '[', ']', '&', '=', '*', '/', '!', '?', '$', '.', '@', '%', 0 };

enum class Decision
{
	Success = 0, Idle = 0, MoveAhead, LayEgg
};

struct Mravenec {

	int16_t posx, posy;
	uint8_t state;
	uint8_t eggGrowth;
	uint8_t irPtr;
	uint8_t memPtr;
	uint8_t energy;
	uint8_t temperature;
	uint8_t progMemory[UNIT_PROGMEM_SIZE];
	uint8_t dynMemory[UNIT_DYNMEM_SIZE];

	void setProgram(const char* prog, bool strict = false);
	__device__ void copyMutatedProgramTo(Mravenec* target, float mutationProb, uint32_t& rand,
		int segmtStart = 0, int segmtEnd = UNIT_PROGMEM_IRCOUNT);
	__both__ inline void setInstruction(uint8_t index, uint8_t instruction)
	{
		int shift = (index & 1) * 4;
		int mask = ~(0xf << shift);
		progMemory[index / 2] = (progMemory[(index % UNIT_PROGMEM_IRCOUNT) / 2] & mask) | (instruction << shift);
	}


	// Instructions are stored 2 per byte.
	__both__ inline IR instructionAt(int index) { return IR((progMemory[mod(index, UNIT_PROGMEM_IRCOUNT) / 2] >> ((index & 1) << 2)) & 15); }
	__both__ inline uint8_t dynmemAt(int index) { return dynMemory[mod(index, UNIT_DYNMEM_SIZE)]; }

	__both__ inline bool alive() { return get_bit(state, 2); }
	__both__ inline void alive(bool value) { set_bit(state, 2, value); }
	__both__ inline uint8_t direction() { return get_bits(state, 0, 2); }
	__both__ inline void direction(int dir) { set_bits(state, 0, 2, dir); }
	__both__ inline Decision decision() { return (Decision)get_bits(state, 3, 3); }
	__both__ inline void decision(Decision value) { set_bits(state, 3, 3, (int)value); }

	__device__ void stepProgram(Level &level);

	void readFrom(std::ifstream& infile);
	void writeTo(std::ofstream& outfile);
	void toString(char* c, int id);
};


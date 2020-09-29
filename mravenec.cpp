#include "mravenec.h"
#include "utils.h"
#include "screen.h"
#include <cstdio>
#include "file_errors.h"

#define dynMemCurrent dynMemory[memPtr]
#define memPtrNext mod(size_t(memPtr) + 1, UNIT_DYNMEM_SIZE)
#define memPtrPrev mod(size_t(memPtr) - 1, UNIT_DYNMEM_SIZE)

__host__ void Mravenec::setProgram(const char* program, bool strict)
{
	if (!program) return;

	char c = *program;
	int ip = 0;

	for (int i = 0; c && ip < UNIT_PROGMEM_IRCOUNT; c = program[++i])
	{
		int ir;
		for (ir = 0; ir <= 16; ir++)
			if (IR_MNEM[ir] == c) break;
		if (ir < 16)
			setInstruction(ip++, ir);
		else if (strict)
		{
			char msg[256];
			sprintf_s(msg, "Unknown instruction %c at position %d.", c, i);
		}

	}
}

__device__ void Mravenec::copyMutatedProgramTo(Mravenec* target, float mutationProb, uint32_t& rand, int segmtStart, int segmtEnd)
{
	if (segmtEnd < segmtStart) segmtEnd += UNIT_PROGMEM_IRCOUNT;
	int dice = (int)(1.f / mutationProb);

	for (int i = segmtStart; i < segmtEnd; i++)
	{
		int instr = int(this->instructionAt(i));
		if (rnd_next(rand) % dice == 0)
			instr = rnd_next(rand) & 0xf;
		target->setInstruction(i, instr);
	}
}

__device__ void Mravenec::stepProgram(Level& level)
{
	auto last_ir = instructionAt(irPtr - 1);
	auto ir = instructionAt(irPtr++);
	Tile& t = *(level.tileAt(posx, posy));
	Tile* ahead = level.tileAt(posx + dx[direction()], posy + dy[direction()]);
	Mravenec* m_ahead = nullptr;
	if (ahead) m_ahead = level.mravenecAt(*ahead);

	// write result from the last instruction
	switch (last_ir)
	{
	case IR::LayEgg:
	case IR::Move:
		if (decision() == Decision::Success)
			dynMemCurrent = 1;
		else
			dynMemCurrent = 0;
		break;
	}

	// do the current instruction
	switch ((IR)ir)
	{
	case IR::Next: memPtr = memPtrNext; break;
	case IR::Prev: memPtr = memPtrPrev; break;
	case IR::Increment: dynMemCurrent++; break;
	case IR::Decrement: dynMemCurrent--; break;
	case IR::ShiftL: dynMemCurrent <<= 1; break;
	case IR::ShiftR: dynMemCurrent >>= 1; break;
	case IR::Add: dynMemCurrent += dynMemory[memPtrPrev]; break;
	case IR::Sub: dynMemCurrent -= dynMemory[memPtrPrev]; break;
	case IR::WhileDo:
		if (!byte_positive(dynMemCurrent))
		{
			int level = 1;
			for (int i = 0; i < UNIT_PROGMEM_IRCOUNT; i++)
				switch (instructionAt(irPtr + i))
				{
				case IR::WhileDo: level++; break;
				case IR::Done:
					level--;
					if (level == 0) { irPtr += mod(+i + 1, UNIT_PROGMEM_IRCOUNT); break; }
					break;
				}
		}
		break;
	case IR::Done:
		if (byte_positive(dynMemCurrent))
		{
			int level = 1;
			for (int i = 2; i <= UNIT_PROGMEM_IRCOUNT; i++)
				switch (instructionAt(irPtr - i))
				{
				case IR::Done: level++; break;
				case IR::WhileDo:
					level--;
					if (level == 0) { irPtr += mod(-i + 1, UNIT_PROGMEM_IRCOUNT); break; }
					break;
				}
		}
		break;
	case IR::Move:
		int movement = dynMemCurrent & 0b11;
		switch (movement)
		{
		case 0: decision(Decision::MoveAhead); break;
		case 1: direction(dir_right(direction())); break;
		case 3: direction(dir_left(direction())); break;
		}
		break;
	case IR::Sense:
		if (!ahead)
			dynMemCurrent = -2;
		else if (instructionAt(irPtr - 1) == IR::Sense)
		{
			if (m_ahead)
				dynMemCurrent = m_ahead->dynmemAt(0);
			else
				dynMemCurrent = ahead->state & 0xf;
		}
		else
		{
			switch (ahead->type)
			{
			case TileType::Void: dynMemCurrent = 1; break;
			case TileType::Food: dynMemCurrent = max(2, (ahead->state & 0xf) >> 1); break;
			case TileType::Mravenec: dynMemCurrent = 0; break;
			case TileType::Egg: dynMemCurrent = -1; break;
			case TileType::Wall: dynMemCurrent = -max(2, (ahead->state & 0xf) >> 1); break;
			}
		}
		break;
	case IR::Measure:
		if (instructionAt(irPtr - 1) == IR::Measure)
			// measure energy
			dynMemCurrent = energy;
		else
			// measure temperature
			dynMemCurrent = temperature;
		break;
	case IR::Burn:
		energy--;
		temperature = min(255, temperature + int_amount(level.config.heating_amount, t.random));
		break;
	case IR::LayEgg:
		decision(Decision::LayEgg);
		break;
	}
}

void Mravenec::readFrom(std::ifstream& infile)
{
	posx = posy = -1;

	int _irPtr, _memPtr, _energy, _heat;

	std::string state_bin, program;
	if (!(infile >> posx >> posy))
		bad_unit(posx, posy, "Position couldn't be read.");
	infile >> state_bin;
	if (!(infile >> _irPtr >> _memPtr >> _energy >> _heat))
		bad_unit(posx, posy, "Check irPtr, memPtr, energy, heat.");
	infile >> program;
	irPtr = _irPtr;
	memPtr = _memPtr;
	energy = _energy;
	temperature = _heat;
	try
	{
		setProgram(program.c_str(), true);
		state = from_binary(state_bin.c_str());
	}
	catch (std::exception e)
	{
		bad_unit(posx, posy, e.what());
	}

	for (int i = 0; i < UNIT_DYNMEM_SIZE; i++)
	{
		int byte;
		if (infile >> byte)
		{
			dynMemory[i] = byte;
		}
		else
		{
			auto msg = std::string("Dynamic memory corrupt, position ") + std::to_string(i + 1);
			bad_unit(posx, posy, msg.c_str());
		}
	}
}

void Mravenec::writeTo(std::ofstream& outfile)
{
	char line[1024];
	sprintf(line, "    " TOK_UNIT " %d %d " BYTE_TO_BINARY_PATTERN " %d %d %d %d", posx, posy, BYTE_TO_BINARY(state), irPtr, memPtr, energy, temperature);
	outfile << line << std::endl << "        ";

	for (int i = 0; i < UNIT_PROGMEM_IRCOUNT; i++)
		outfile << IR_MNEM[(int)instructionAt(i)];

	outfile << std::endl << "       ";
	for (int i = 0; i < UNIT_DYNMEM_SIZE; i++)
		outfile << " " << (int)dynMemory[i];

	outfile << std::endl;
}

void Mravenec::toString(char* c, int id)
{
	sprintf(c, "#%03d:%c at (%2d,%2d) E=%3d s=" BYTE_TO_BINARY_PATTERN " ip=%3d mp=%3d   ",
		id, DIRECTION_CHAR[direction()], posx, posy, energy, BYTE_TO_BINARY(state), irPtr, memPtr);
}

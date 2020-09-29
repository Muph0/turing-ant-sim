
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>

#include "sim.h"
#include "level.h"
#include "file_errors.h"

void Config::readFrom(std::ifstream& infile)
{
	std::string token;
	float value = 0;
	while (infile >> token && token != TOK_END_BLOCK)
	{
		infile >> value;

		try_set_symbol(food_growth_speed);
		else try_set_symbol(food_growth_step);
		else try_set_symbol(food_growth_max);
		else try_set_symbol(movement_cost);
		else try_set_symbol(heating_amount);
		else try_set_symbol(energy_death_th);
		else try_set_symbol(temp_death_th);
		else try_set_symbol(ambient_temp);
		else try_set_symbol(egg_laying_th);
		else try_set_symbol(egg_growth);
		else try_set_symbol(ground_conductivity);
		else try_set_symbol(wall_conductivity);
		else try_set_symbol(ant_conductivity);
		else unknown_token(token);
	}
}

void Config::writeTo(std::ofstream& outfile)
{
#define output_symbol(SYMBOL) (outfile << "    " << setw(28) << left << #SYMBOL ":" << SYMBOL << endl) 
	using namespace std;
	output_symbol(food_growth_speed);
	output_symbol(food_growth_step);
	output_symbol(food_growth_max);
	output_symbol(movement_cost);
	output_symbol(heating_amount);
	output_symbol(energy_death_th);
	output_symbol(temp_death_th);
	output_symbol(ambient_temp);
	output_symbol(egg_laying_th);
	output_symbol(egg_growth);
	output_symbol(ground_conductivity);
	output_symbol(wall_conductivity);
	output_symbol(ant_conductivity);
}

// Returns nullptr if the copy has no ant on it.
__both__ Mravenec* Level::mravenecAt(Tile& t)
{
	return t.type == TileType::Mravenec || t.type == TileType::Egg ? mravenci + t.state : nullptr;
}
// Returns nullptr if the index is out of bounds.
__both__ Mravenec* Level::mravenecAt(int x, int y)
{
	return inside(x, y) ? mravenecAt(tiles[x + y * width]) : nullptr;
}
// Returns nullptr if the index is out of bounds.
__both__ Tile* Level::tileAt(int x, int y)
{
	return inside(x, y) ? &tiles[x + y * width] : nullptr;
}

Level::Level(int width, int height, int unitCap)
{
	this->width = width;
	this->height = height;
	this->unitCap = unitCap;

	cudaMallocManaged(&tiles, size() * sizeof(Tile));
	cudaMallocManaged(&mravenci, unitCap * sizeof(Mravenec));
	cudaMallocManaged(&freeUnitList, unitCap * sizeof(size_t));
}

__both__ void Level::freeUnit(Mravenec* m)
{
	freeUnitList[freeUnitsCount++] = m;
	m->alive(false);
}
__both__ Mravenec* Level::allocUnit()
{
	if (freeUnitsCount > 0)
	{
		return freeUnitList[--freeUnitsCount];
	}

	return nullptr;
}

void Level::readUnitsFrom(std::ifstream& infile)
{
	std::string token;

	while (infile >> token && token != TOK_END_BLOCK)
	{
		Mravenec* m = allocUnit();
		if (token == TOK_UNIT)
		{
			m->readFrom(infile);
			Tile* t = tileAt(m->posx, m->posy);
			t->state = int(m - mravenci);
			t->type = TileType::Mravenec;
		}
		else
		{
			expect_token(token, TOK_UNIT);
		}
	}
}
void Level::writeUnitsTo(std::ofstream& outfile)
{
	for (int i = 0; i < unitCap; i++)
	{
		Mravenec& m = mravenci[i];
		if (m.alive())
			m.writeTo(outfile);
	}
}

void Level::readTilesFrom(std::ifstream& infile)
{
	using namespace std;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < height; x++)
		{
			Tile copy = {};
			uint64_t& input = (uint64_t&)copy;
			infile >> hex >> input;

			Tile* target = &(tiles[x + y * width]);

			// in case ants are read before tiles
			// so we dont overwrite them
			if (target->type == TileType::Mravenec || target->type == TileType::Egg)
			{
				copy.state = target->state;
				copy.type = target->type;
			}

			*target = copy;
		}
	}

	string token;
	infile >> token;
	expect_token(token, TOK_END_BLOCK);
}

void Level::writeTilesTo(std::ofstream& ofile)
{
	using namespace std;

	for (int y = 0; y < height; y++)
	{
		ofile << "    ";
		for (int x = 0; x < height; x++)
		{
			Tile copy = *(tileAt(x, y));

			// to allow ant movement by the user
			if (copy.type == TileType::Mravenec || copy.type == TileType::Egg)
			{
				copy.type = TileType::Void;
				copy.state = 0;
			}

			uint64_t& out = (uint64_t&)copy;
			ofile << setfill('0') << setw(16) << hex << out << " ";
		}
		ofile << endl;
	}
}

void Level::initTiles()
{
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			Tile t;
			t.type = TileType::Void;
			t.state = 0;
			t.temperature = config.ambient_temp;
			t.random = rand() << 16 | rand();

			tiles[x + y * width] = t;
		}
	}
}
void Level::initUnits()
{
	memset(mravenci, 0, sizeof(Mravenec) * unitCap);
	for (freeUnitsCount = 0; freeUnitsCount < unitCap; freeUnitsCount++)
		freeUnitList[freeUnitsCount] = &mravenci[unitCap - freeUnitsCount - 1];
}
void Level::spawnRandomUnits(int count)
{
	for (int i = 0; i < count && i < unitCap; i++)
	{
		Mravenec& u = *(allocUnit());

		do {
			u.posx = rand() % width;
			u.posy = rand() % height;
		} while (tileAt(u.posx, u.posy)->type != TileType::Void);


		u.state = 0b100;
		u.direction(rand() % 4);
		u.alive(true);

		u.irPtr = 0;
		u.memPtr = 0;
		u.energy = 128;
		u.temperature = 128;

		for (int n = 0; n < UNIT_PROGMEM_SIZE; n++)
			u.progMemory[n] = rand();
		for (int n = 0; n < UNIT_DYNMEM_SIZE; n++)
			u.dynMemory[n] = 0;

		mravenci[i] = u;
		Tile& t = tiles[u.posx + u.posy * width];

		t.type = TileType::Mravenec;
		t.state = i;
	}
}
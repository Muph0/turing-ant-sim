#pragma once

#include <cuda_runtime.h>

class Level;

#include "utils.h"
#include "tile.h"
#include "mravenec.h"

struct Config
{
	void readFrom(std::ifstream& infile);
	void writeTo(std::ofstream& outfile);

	float food_growth_speed = 0.0001;
	float food_growth_step = 10;
	float food_growth_max = 80;
	float movement_cost = 0.1;
	float heating_amount = 10;
	float energy_death_th = 0;
	float temp_death_th = 32;
	float ambient_temp = 10;
	float egg_laying_th = 50;
	float egg_growth = 0.1;
	float gene_change_prob = 2.f / UNIT_PROGMEM_IRCOUNT;

	float ground_conductivity = 1;
	float wall_conductivity = 0.001;
	float ant_conductivity = 0.1;
};

class Level
{
private:
	Mravenec** freeUnitList;

public:
	Config config;
	Mravenec* mravenci;
	Tile* tiles;
	int width, height, unitCap;
	int freeUnitsCount;
	__both__ inline int size() { return width * height; }
	__both__ inline bool inside(int x, int y) { return x >= 0 && x < width&& y >= 0 && y < height; }

	// Returns nullptr if the tile has no ant on it.
	__both__ Mravenec* mravenecAt(Tile& t);
	// Returns nullptr if the index is out of bounds.
	__both__ Mravenec* mravenecAt(int x, int y);
	// Returns nullptr if the index is out of bounds.
	__both__ Tile* tileAt(int x, int y);

	Level(int width, int height, int unitCount);

	void initUnits();
	void initTiles();
	void readUnitsFrom(std::ifstream& infile);
	void writeUnitsTo(std::ofstream& ofile);
	void readTilesFrom(std::ifstream& infile);
	void writeTilesTo(std::ofstream& ofile);

	void spawnRandomUnits(int count);

	__both__ void freeUnit(Mravenec* m);
	__both__ Mravenec* allocUnit();
};
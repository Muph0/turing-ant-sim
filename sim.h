#pragma once

#include <cstdint>
#include <fstream>
#include "level.h"
#include "mravenec.h"
#include "tile.h"

#define TOK_MAGIC			"turing_ant_simulation_file"
#define TOK_PARAMS			"params"
#define TOK_BEGIN_BLOCK		"{"
#define TOK_END_BLOCK		"}"
#define TOK_UNIT_DATA		"unit_data"
#define TOK_TILE_DATA		"tile_data"
#define TOK_UNIT			"unit"

constexpr int SIM_BLOCK_WIDTH = 32, SIM_BLOCK_SIZE = 32 * 32;
constexpr int POPULATION_CAP = 1 << 6;

enum class UIState
{
	Simulation
};
enum class Backdrop
{
	None, RandomState, Heatmap, Feromone
};

class Simulation
{
private:
	uint64_t _ticks = 0;
	uint64_t _startedMs = 0;
	uint64_t _lastFrameMs = 0;
	int _frameDuration = 0;

	const char* _filename;

	UIState uistate = UIState::Simulation;

	int _speed = 0;
	bool _doStep = false;

	float _panX = 0.0f;
	float _panY = 0.0f;

	Level* _level = nullptr;

	Mravenec* _target = nullptr;
	int _movePenalty = 1;

	void handleUserInput();

	void speedUp();
	void slowDown();
	void doStep();

	void panX(int delta);
	void panY(int delta);
	void panStepX(int delta);
	void panStepY(int delta);
	void panRound();

	// gets called once per Simulation tick
	void tick(const dim3 blocks, const dim3 threads, const int unitCount);

	// gets called once per user frame
	void update();

public:
	inline Level& level() { return *_level; }
	inline const char* filename() { return _filename; }

	Simulation(const char* filename);
	~Simulation();

	int panX();
	int panY();
	bool cursorAt(int x, int y);
	Backdrop backdrop = Backdrop::Heatmap;

	uint64_t ticksElapsed();
	uint64_t micros();
	int ticksPerFrame();
	int frameDuration();
	void waitForFrame();

	Tile* selectedTile();
	Mravenec* selectedUnit();

	void target(Mravenec* target_) { _target = target_; }
	Mravenec* target() { return _target; }

	int tilesMemory();
	int unitsMemory();

	void init(int levelW, int levelH, int unitCap);

	void reloadConfig(std::ifstream& infile);
	void loadFromFile(std::ifstream& infile);
	void writeToFile(std::ofstream& outfile);

	void run();

};

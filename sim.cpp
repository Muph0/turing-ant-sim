#include <chrono>
#include <thread>
#include <iostream>
#include <iomanip>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "utils.h"
#include "screen.h"
#include "sim.h"
#include "file_errors.h"

int Simulation::ticksPerFrame() { return _speed > 0 ? 1 << (_speed - 1) : 0; }
uint64_t Simulation::ticksElapsed() { return this->_ticks; }
uint64_t Simulation::micros()
{
	using namespace std::chrono;
	microseconds ms = duration_cast<microseconds>(system_clock::now().time_since_epoch());
	if (_startedMs == 0) _startedMs = ms.count();
	return ms.count() - _startedMs;
}


constexpr int _TARGET_FTIME = 16666;
int Simulation::frameDuration() { return _frameDuration; }
void Simulation::waitForFrame()
{
	std::this_thread::sleep_for(std::chrono::microseconds(_TARGET_FTIME - (micros() - _lastFrameMs)));
}

void Simulation::speedUp() { _speed++; }
void Simulation::slowDown() { if (_speed > 0) _speed--; }
void Simulation::doStep() { _doStep = true; }


static constexpr float panSpeed = 0.1f / _TARGET_FTIME;

void Simulation::panX(int delta) { _panX = clamp(_panX + delta * panSpeed * _frameDuration, 0.5f, _level->width - 0.5f); }
void Simulation::panY(int delta) { _panY = clamp(_panY + delta * panSpeed * _frameDuration, 0.5f, _level->height - 0.5f); }
void Simulation::panStepX(int delta) { _panX = clamp(_panX + delta * 0.5f, 0.0f, float(_level->width - 1)); }
void Simulation::panStepY(int delta) { _panY = clamp(_panY + delta * 0.5f, 0.0f, double(_level->height - 1)); }
int Simulation::panX() { return int(_panX); }
int Simulation::panY() { return int(_panY); }
void Simulation::panRound() { _panX = panX() + 0.5f; _panY = panY() + 0.5f; }
bool Simulation::cursorAt(int x, int y) { return (panX() == x && panY() == y); }

Tile* Simulation::selectedTile() { return _level->tiles + (panX() + panY() * _level->width); }
Mravenec* Simulation::selectedUnit() { Tile* t = selectedTile();  return _level->mravenecAt(*t); }

int Simulation::tilesMemory() { return _level->size() * sizeof(Tile); }
int Simulation::unitsMemory() { return (POPULATION_CAP - _level->freeUnitsCount) * sizeof(Mravenec); }



void Simulation::init(int levelWidth, int levelHeight, int unitCap)
{
	cudaMallocManaged(&_level, sizeof(Level));
	new (_level) Level(levelWidth, levelHeight, unitCap);

	_level->initTiles();
	_level->initUnits();

	update();
}

void Simulation::reloadConfig(std::ifstream& infile)
{
	std::string token;
	while (infile >> token)
	{
		if (token == TOK_PARAMS)
		{
			infile >> token;
			expect_token(token, TOK_BEGIN_BLOCK);
			_level->config.readFrom(infile);
		}
	}
}
void Simulation::loadFromFile(std::ifstream& infile)
{
	std::string token;

	infile >> token;
	if (token != TOK_MAGIC)
		throw std::exception("Bad file format.");

	int width = -1;
	int height = -1;
	int max_units = -1;

	while (infile >> token)
	{
		if (token == "width:") infile >> width;
		else if (token == "height:") infile >> height;
		else if (token == "max_units:") infile >> max_units;
		else if (token == "ticks:") infile >> _ticks;
		else if (token == TOK_PARAMS)
		{
			if (width == -1 || height == -1 || max_units == -1)
				throw std::exception("Bad order of parameters: set width, height and max_units first.");

			infile >> token;
			expect_token(token, TOK_BEGIN_BLOCK);

			this->init(width, height, max_units);
			_level->config.readFrom(infile);
		}
		else if (token == TOK_UNIT_DATA)
		{
			infile >> token;
			expect_token(token, TOK_BEGIN_BLOCK);
			_level->readUnitsFrom(infile);
		}
		else if (token == TOK_TILE_DATA)
		{
			infile >> token;
			expect_token(token, TOK_BEGIN_BLOCK);
			_level->readTilesFrom(infile);
		}
		else
		{
			throw std::exception("Unexpected token.");
		}
	}
}
void Simulation::writeToFile(std::ofstream& ofile)
{
	using namespace std;
	ofile << TOK_MAGIC << endl;
	ofile << setfill(' ') << setw(12) << left << "width:" << _level->width << endl;
	ofile << setfill(' ') << setw(12) << left << "height:" << _level->height << endl;
	ofile << setfill(' ') << setw(12) << left << "max_units:" << _level->unitCap << endl;
	ofile << setfill(' ') << setw(12) << left << "ticks:" << _ticks << endl;
	ofile << endl << TOK_PARAMS << " " TOK_BEGIN_BLOCK << endl;
	_level->config.writeTo(ofile);
	ofile << TOK_END_BLOCK << endl << endl;
	ofile << TOK_UNIT_DATA << " " TOK_BEGIN_BLOCK << endl;
	_level->writeUnitsTo(ofile);
	ofile << TOK_END_BLOCK << endl << endl;
	ofile << TOK_TILE_DATA " " TOK_BEGIN_BLOCK << endl;
	_level->writeTilesTo(ofile);
	ofile << TOK_END_BLOCK << endl;
}
Simulation::Simulation(const char* filename)
{
	_filename = filename;
}
Simulation::~Simulation()
{
	if (_level != nullptr)
	{
		delete _level;
		cudaFree(&_level);
		_level = nullptr;
	}
}


// gets called once per user frame
void Simulation::update()
{
	if (_target != nullptr)
	{
		if (!_target->alive())
		{
			_target = nullptr;
		}
		else
		{
			_panX = _target->posx + 0.5f;
			_panY = _target->posy + 0.5f;
		}
	}

	_frameDuration = micros() - _lastFrameMs;
	_lastFrameMs = micros();

	handleUserInput();
}

void Simulation::run()
{
	initScreen();
	_panX = _level->width / 2;
	_panY = _level->height / 2;

	while (true)
	{
		dim3 tile_blocks((_level->width + 31) / 32, (_level->height + 31) / 32, 1);
		dim3 tile_threads(32, 32, 1);
		int unitCount = _level->unitCap;


		switch (uistate)
		{
		case UIState::Simulation:
			for (int r = 0; r < ticksPerFrame() || _doStep; r++)
			{
				tick(tile_blocks, tile_threads, unitCount);
				_doStep = false;
			}

			gpuErrchk(cudaDeviceSynchronize());

			update();

			clearAndResizeBuffer(FG_GRAY | BG_BLACK);
			drawLevel(*this);
			drawStats(*this);
			drawHotkeys(*this);
			drawFlush();
			break;
		}
		waitForFrame();
	}
}

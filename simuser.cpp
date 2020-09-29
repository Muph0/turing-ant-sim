#include <string>
#include <fstream>

#include "sim.h"
#include "keyboard.h"

void Simulation::handleUserInput()
{
	keyboardUpdate();

	bool allowStep = ticksPerFrame() == 0;
	bool allowPan = target() == nullptr;

	if (isKeyPressed(VK_RETURN))
	{
		if (allowStep)
			doStep();
		else
			_speed = 0;
	}

	if (isKeyPressed(VK_ADD))
		speedUp();
	if (isKeyPressed(VK_SUBTRACT))
		slowDown();

	int panDelta = 1;
	if (isKeyDown('A')) panDelta = 10;

	if (allowPan)
	{
		if (isKeyDown(VK_UP))
			panY(-panDelta);
		if (isKeyDown(VK_DOWN))
			panY(panDelta);
		if (isKeyDown(VK_LEFT))
			panX(-panDelta);
		if (isKeyDown(VK_RIGHT))
			panX(panDelta);

		if (isKeyPressed(VK_UP))
			panStepY(-1);
		if (isKeyPressed(VK_DOWN))
			panStepY(1);
		if (isKeyPressed(VK_LEFT))
			panStepX(-1);
		if (isKeyPressed(VK_RIGHT))
			panStepX(1);

		if (isKeyReleased(VK_UP) || isKeyReleased(VK_DOWN) || isKeyReleased(VK_LEFT) || isKeyReleased(VK_RIGHT))
			panRound();
	}

	if (isKeyPressed('L'))
	{
		auto selected = selectedUnit();

		if (target() != nullptr) target(nullptr);
		else if (selected != nullptr) target(selected);
	}

	if (isKeyPressed('W'))
	{
		std::string filename = _filename;
		size_t i = filename.find_last_of('.');
		std::ofstream ofile(filename.substr(0,i) + "." + std::to_string(ticksElapsed()));
		writeToFile(ofile);
		ofile.close();
	}

	if (isKeyPressed('R'))
	{
		std::ifstream infile(_filename);
		reloadConfig(infile);
		infile.close();
	}

#define bind_key_to_backdrop(KEY, BACKDROP) do { \
	if (isKeyPressed(KEY)) { if (backdrop == (BACKDROP)) backdrop = Backdrop::None; else backdrop = (BACKDROP); } } while (0)

	bind_key_to_backdrop('H', Backdrop::Heatmap);
	bind_key_to_backdrop('G', Backdrop::RandomState);
	bind_key_to_backdrop('F', Backdrop::Feromone);
}


#include <iostream>
#include <fstream>
#include "sim.h"


int main(int argc, const char const* argv[])
{
	if (argc < 2)
	{
		std::cout << "Pass a simulation file as first argument." << std::endl;
		return 1;
	}

	std::ifstream infile(argv[1]);
	Simulation simulation(argv[1]);

	if (infile.good())
	{
		simulation.loadFromFile(infile);
		infile.close();
	}
	else
	{
		infile.close();
		std::cout << "Creating a new simulation. Please enter the level width, height and the unit cap." << std::endl;
		int width, height, unitCap, startPopulation;
		std::cin >> width >> height >> unitCap;
		std::cout << "OK, initializing... ";
		simulation.init(width, height, unitCap);
		std::cout << "done. Please enter the size of the starting population." << std::endl;
		std::cin >> startPopulation;
		simulation.level().spawnRandomUnits(startPopulation);

		std::ofstream outfile(simulation.filename());
		simulation.writeToFile(outfile);
		outfile.close();
	}

	simulation.run();
}
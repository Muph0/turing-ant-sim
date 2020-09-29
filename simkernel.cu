
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "sim.h"
#include "utils.h"
#include "mravenec.h"

__global__ void sim_tickProgram(Level& level);
__global__ void sim_updateTile(Level& level);

// gets called once per Simulation tick
void Simulation::tick(const dim3 blocks, const dim3 threads, const int unitCount)
{
	_doStep = false;

	// sequentially called CUDA kernels will be executed in series
	// _level is in managed memory, so thats okay too
	if (unitCount > 0)
		sim_tickProgram __kernel_call((unitCount + 511) / 512, 512) (*_level);

	sim_updateTile __kernel_call(blocks, threads) (*_level);

	_ticks++;
}

__global__ void sim_tickProgram(Level& level)
{
	int mravenec_i = blockIdx.x * 512 + threadIdx.x;
	if (mravenec_i >= 0 && mravenec_i < level.unitCap)
	{
		Mravenec& mravenec = level.mravenci[mravenec_i];

		if (mravenec.alive())
			mravenec.stepProgram(level);
	}
}

__global__ void sim_updateTile(Level& level)
{
	int X = blockIdx.x * 32 + threadIdx.x;
	int Y = blockIdx.y * 32 + threadIdx.y;

	const float dT = 0.010;

	Tile* tile_ptr = level.tileAt(X, Y);
	Tile tile;

	if (tile_ptr != nullptr)
	{
		tile = *tile_ptr;
		Mravenec *m = level.mravenecAt(tile);

		if (m != nullptr)
		{
			if (m->posx == X && m->posy == Y)
			{
				// conduct temperature from the ant to the tile its standing on
				float a = sqrt(level.config.ant_conductivity * level.config.ground_conductivity);
				int dQ = int_amount(a * dT * (tile.temperature - m->temperature), tile.random);
				m->temperature += dQ;
				tile.temperature -= dQ;
			}
		}

		float local_cond = 0.f;
		switch (tile.type)
		{
		case TileType::Mravenec:
		case TileType::Food:
		case TileType::Void: local_cond = level.config.ground_conductivity; break;
		case TileType::Wall: local_cond = level.config.wall_conductivity; break;
		}

		// temperature conductivity
		{
			float dQ = 0.0f;
			for (int i = 0; i < 4; i++)
			{
				Tile* t = level.tileAt(X + dx[i], Y + dy[i]);
				float neighbor_conductivity = level.config.ground_conductivity;
				int neighbor_temp = level.config.ambient_temp;

				if (t != nullptr)
				{
					neighbor_temp = t->temperature;
					if (t->type == TileType::Wall)
					{
						neighbor_conductivity = level.config.wall_conductivity;
					}
				}

				dQ += sqrt(local_cond * neighbor_conductivity) * dT * (neighbor_temp - tile.temperature);
			}

			tile.temperature += int_amount(dQ, tile.random);
		}

		// food growth		
		{
			int grow = int_amount(level.config.food_growth_speed, tile.random);
			int step = (int)level.config.food_growth_step;
			if (grow != 0)
			{
				if (tile.type == TileType::Void)
				{
					tile.type = TileType::Food;
					tile.state = 0;
				}

				if (tile.type == TileType::Food && tile.state < level.config.food_growth_max)
					tile.state += step <= 1 ? step : step / 2 + (rnd_next(tile.random) % step);
			}
		}

		// egg growth
		if (tile.type == TileType::Egg)
		{
			int growth = int_amount(level.config.egg_growth, tile.random);
			if (int(m->eggGrowth) + growth > 255)
				tile.type = TileType::Mravenec;
			else
				m->eggGrowth += growth;
		}

		// movement
		//if (tile.type == TileType::Void || tile.type == TileType::Food)
		{
			Mravenec* want_go_here[] = { 0, 0, 0, 0 };
			int wgh_count = 0;

			// see, who wants to interfere with this tile
			for (int d = 0; d < 4; d++)
			{
				Tile* sousedni = level.tileAt(X + dx[d], Y + dy[d]);
				if (sousedni && sousedni->type == TileType::Mravenec)
				{
					Mravenec* m = level.mravenecAt(*sousedni);
					if (m && dir_back(d) == m->direction())
						switch (m->decision())
						{
						case Decision::LayEgg:
						case Decision::MoveAhead:
							want_go_here[d] = m;
							wgh_count++;
							break;
						}
				}
			}

			if (wgh_count > 0)
			{
				// decide on one
				int winner_i = 1 + (rnd_next(tile.random) % wgh_count);
				int d = 0;
				for (; winner_i; d++)
					if (want_go_here[d] != nullptr)
						winner_i--;

				Mravenec& winner = *(want_go_here[d - 1]);

				// perform their decided action
				switch (winner.decision())
				{
				case Decision::MoveAhead:
					switch (tile.type)
					{
					case TileType::Food:
						winner.energy = min(255, winner.energy + (int(tile.state) & 0xff));
						tile.type = TileType::Void;
						no_break;
					case TileType::Void:
						tile.type = TileType::Mravenec;
						tile.state = &winner - level.mravenci;
						winner.posx = X;
						winner.posy = Y;
						winner.decision(Decision::Success);
						break;
					}
					break;
				case Decision::LayEgg:
					switch (tile.type)
					{
					case TileType::Void:
						Mravenec* egg = level.allocUnit();
						if (!egg) break;
						char* egg_start = (char*)egg;
						for (int i = 0; i < sizeof(Mravenec); i++)
							egg_start[i] = 0;

						tile.type = TileType::Egg;
						winner.copyMutatedProgramTo(egg, level.config.gene_change_prob, tile.random);
						egg->posx = X;
						egg->posy = Y;
						egg->alive(true);
						egg->temperature = winner.temperature / 2;
						winner.temperature /= 2;
						int transfer_e = min(winner.dynmemAt(winner.memPtr), winner.energy);
						egg->energy = transfer_e;
						winner.energy -= transfer_e;
						tile.state = egg - level.mravenci;
						break;
					case TileType::Egg:
						int seg_start = rnd_next(tile.random) % UNIT_PROGMEM_IRCOUNT;
						int seg_end = rnd_next(tile.random) % UNIT_PROGMEM_IRCOUNT;
						winner.copyMutatedProgramTo(m, level.config.gene_change_prob, tile.random, seg_start, seg_end);
						break;
					}
					break;
				default:
					break;
				}

				// leave others be - no interaction is to be interpreted as a failure
			}
		}

	}

	__syncthreads();

	if (tile_ptr)
	{
		Mravenec* m = level.mravenecAt(tile);

		// move out condition
		if (tile.type == TileType::Mravenec &&
			m && (m->posx != X || m->posy != Y))
		{
			tile.state = m->dynmemAt(0);
			tile.type = TileType::Void;
		}

		// death condition
		if ((tile.type == TileType::Mravenec || tile.type == TileType::Egg) &&
			m && (m->energy <= level.config.energy_death_th || m->temperature <= level.config.temp_death_th))
		{
			tile.state = 1 << 8 | m->energy;
			tile.type = TileType::Food;
			level.freeUnit(m);
		}

		*tile_ptr = tile;
	}


	//printf("exit ");
}


#include "screen.h"
#include <cstdio>
#include "keyboard.h"

#define screen_char(X,Y) (chiBuffer[(X) + (Y) * screenWidth].Char.AsciiChar)
#define screen_attr(X,Y) (chiBuffer[(X) + (Y) * screenWidth].Attributes)

int screenWidth = 80;
int screenHeight = 60;

constexpr int STATS_HEIGHT = 12;

HANDLE screenBuffer;
CHAR_INFO* chiBuffer;

constexpr CHAR_INFO char_info(char chr, uint16_t attr)
{
	CHAR_INFO chi = {};
	chi.Attributes = attr;
	chi.Char.AsciiChar = chr;
	return chi;
}

void resizeScreen(int width, int height)
{
	if (chiBuffer)
	{
		delete[] chiBuffer;
		chiBuffer = nullptr;
	}

	screenWidth = width;
	screenHeight = height;

	chiBuffer = new CHAR_INFO[screenWidth * screenHeight];
}
bool resizeIfNeeded()
{
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
	int width = csbi.srWindow.Right - csbi.srWindow.Left + 1;
	int height = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;

	if (width != screenWidth || height != screenHeight)
	{
		resizeScreen(width, height);
		return true;
	}

	return false;
}

void initScreen()
{
	screenBuffer = GetStdHandle(STD_OUTPUT_HANDLE);

	CONSOLE_CURSOR_INFO info;
	GetConsoleCursorInfo(screenBuffer, &info);
	info.bVisible = false;
	SetConsoleCursorInfo(screenBuffer, &info);
}

void clearAndResizeBuffer(WORD attributes)
{
	resizeIfNeeded();
	for (int i = 0; i < screenWidth * screenHeight; i++)
	{
		chiBuffer[i].Char.AsciiChar = 0;
		chiBuffer[i].Attributes = attributes;
	}
}

void hotkey(int& x, char hotkey, char* tip, uint8_t color, uint8_t inverted, bool invert, uint8_t highlight)
{
	int y = screenHeight - 1;
	hotkey = uppercase(hotkey);

	drawString(tip, x, y, invert ? inverted : color);
	for (int i = 0; tip[i]; i++)
	{
		char upper = uppercase(tip[i]);
		if (upper == hotkey)
		{
			//screen_char(x, y) = upper;
			screen_attr(x + i, y) = highlight | (invert ? inverted : 0);
			break;
		}
	}
	x += strlen(tip);
	drawString(" | ", x, y, color);
	x += 3;
}
void drawHotkeys(Simulation& sim)
{
	uint8_t color = BG_BLUE | FG_YELLOW | FG_BRIGHT;
	uint8_t	inverted = color >> 4 | color << 4;
	uint8_t highlight = BG_BLUE | FG_RED | FG_BRIGHT;

	for (int i = 0; i < screenWidth; i++)
		screen_attr(i, screenHeight - 1) = color;

	int x = 1;
	hotkey(x, 'A', "Fast pan", color, inverted, isKeyDown('A'), highlight);
	hotkey(x, 'L', "Lock", color, inverted, sim.target() != nullptr, highlight);
	hotkey(x, 'W', "Write snapshot", color, inverted, isKeyDown('W'), highlight);
	drawString("SHOW: ", x, screenHeight - 1, color); x += 6;
	hotkey(x, 'F', "Feromones", color, inverted, sim.backdrop == Backdrop::Feromone, highlight);
	hotkey(x, 'G', "Rng state", color, inverted, sim.backdrop == Backdrop::RandomState, highlight);
	hotkey(x, 'H', "Heatmap", color, inverted, sim.backdrop == Backdrop::Heatmap, highlight);
}

void drawFlush()
{
	COORD bufSize, COORD_ZERO;
	SMALL_RECT targetRect;
	bufSize.X = screenWidth;
	bufSize.Y = screenHeight;
	targetRect.Left = 0;
	targetRect.Top = 0;
	targetRect.Right = screenWidth + targetRect.Left;
	targetRect.Bottom = screenHeight + targetRect.Top;
	COORD_ZERO.X = 0;
	COORD_ZERO.Y = 0;

	WriteConsoleOutput(screenBuffer, chiBuffer, bufSize, COORD_ZERO, &targetRect);
	//SetConsoleCursorPosition(screenBuffer, COORD_ZERO);
}

constexpr int HEAT_COLORS[] = {
	FG_BLUE,
	FG_BLUE | FG_BRIGHT ,
	FG_CYAN,
	FG_CYAN | FG_BRIGHT ,
	FG_GREEN,
	FG_GREEN | FG_BRIGHT ,
	FG_YELLOW,
	FG_YELLOW | FG_BRIGHT,
	FG_RED,
	FG_RED | FG_BRIGHT,
	FG_MAGENTA,
	FG_MAGENTA | FG_BRIGHT,
	FG_WHITE
};

CHAR_INFO heat_char(int heatAmt)
{
	heatAmt = min(255, max(0, heatAmt));

	int stepsPerColor = 256 / 12;
	int step = (heatAmt % stepsPerColor) * 6 / stepsPerColor;

	int color1 = heatAmt / stepsPerColor;
	int color2 = min(color1 + 1, _countof(HEAT_COLORS) - 1);

	if (color1 == color2)
		step = 0;

	switch (step)
	{
	default:
	case 0: return char_info(' ', HEAT_COLORS[color1] << 4);
	case 1: return char_info(176, HEAT_COLORS[color1] << 4 | HEAT_COLORS[color2]);
	case 2: return char_info(177, HEAT_COLORS[color1] << 4 | HEAT_COLORS[color2]);
	case 3: return char_info(178, HEAT_COLORS[color2] << 4 | HEAT_COLORS[color1]);
	case 4: return char_info(177, HEAT_COLORS[color2] << 4 | HEAT_COLORS[color1]);
	case 5: return char_info(176, HEAT_COLORS[color2] << 4 | HEAT_COLORS[color1]);
	}
}

const CHAR_INFO GRASS_CHARS[] = {
	char_info(176, FG_GREEN),
	char_info(177, FG_GREEN),
	char_info(178, FG_GREEN),
	char_info(176, FG_GREEN | FG_BRIGHT | BG_GREEN),
	char_info(177, FG_GREEN | FG_BRIGHT | BG_GREEN),
	char_info(178, FG_GREEN | FG_BRIGHT | BG_GREEN),
};
CHAR_INFO grass_char(int grassEnergy)
{
	return GRASS_CHARS[min(grassEnergy * _countof(GRASS_CHARS) / 256, _countof(GRASS_CHARS))];
}

void drawLevel(Simulation& sim)
{
	Level& level = sim.level();

	for (int sy = 0; sy < screenHeight - STATS_HEIGHT; sy++)
		for (int sx = 0; sx < screenWidth; sx++)
		{
			int x = sx + sim.panX() - screenWidth / 2;
			int y = sy + sim.panY() - (screenHeight - STATS_HEIGHT) / 2;

			Tile* tile = level.tileAt(x, y);
			if (tile)
			{
				Mravenec* m = level.mravenecAt(*tile);
				int fg = 0, bg = 0;

				switch (tile->type)
				{
				case TileType::Mravenec:
					screen_char(sx, sy) = DIRECTION_CHAR[m->direction()];
					if (sim.backdrop != Backdrop::Feromone)
						screen_attr(sx, sy) = BG_RED | BG_BRIGHT;
					else
					{
						fg = m->dynmemAt(0) & 0xf;
						bg = m->dynmemAt(0) >> 4 & 0xf;
						if ((sim.micros() >> 17 & 1) && fg == 0) fg = FG_WHITE;
						if (!(sim.micros() >> 17 & 1) && bg == 0) bg = FG_WHITE;
						screen_attr(sx, sy) = fg | bg << 4;
					}
					break;
				case TileType::Egg:
					screen_char(sx, sy) = 'O';
					screen_attr(sx, sy) = FG_YELLOW | FG_BRIGHT;
					break;
				case TileType::Food:
					chiBuffer[sx + sy * screenWidth] = grass_char(tile->state & 0xff);
					if (tile->state >> 8 == 1)
					{
						screen_char(sx, sy) = 'x';
						screen_attr(sx, sy) = FG_GREEN;
						break;
					}
					no_break;
				default:
					switch (sim.backdrop)
					{
					case Backdrop::RandomState:
						screen_char(sx, sy) = (tile->random % 10) + '0';
						screen_attr(sx, sy) = FG_BLUE;
						break;
					case Backdrop::Heatmap:
						chiBuffer[sx + sy * screenWidth] = heat_char(tile->temperature);
						break;
					case Backdrop::Feromone:
						screen_char(sx, sy) = 178;
						screen_attr(sx, sy) = tile->state & 0xf;
						break;
					}
					break;
				}

				if (sim.cursorAt(x, y))
				{
					screen_attr(sx, sy) = ~screen_attr(sx, sy);
				}
			}
			else
			{
				screen_attr(sx, sy) = BG_BLUE;
			}
		}

	if (sim.backdrop == Backdrop::Heatmap)
		for (int sy = 0; sy < _countof(HEAT_COLORS); sy++)
		{
			int sx = screenWidth - 5;
			int temp = 256 / _countof(HEAT_COLORS) * sy;
			char buf[6];
			sprintf_s(buf, "%5d", temp);
			drawString(buf, sx, sy, HEAT_COLORS[sy] << 4);
		}

}

void drawString(char* str, int posx, int posy, WORD attributes = 0x07)
{
	int x = posx;
	int y = posy;

	if (posx < 0 || posx >= screenWidth || posy < 0 || posy > screenHeight) return;

	for (int i = 0; str[i] > 0; i++)
	{
		if (str[i] != '\n')
		{
			chiBuffer[x + y * screenWidth].Char.AsciiChar = str[i];
			chiBuffer[x + y * screenWidth].Attributes = attributes;
		}

		x++;
		if (x > screenWidth || str[i] == '\n') {
			y++;
			x = posx;
		}
		if (y > screenHeight) {
			y = posx; x = posy;
		}
	}
}

void drawStats(Simulation& sim, bool justTime)
{
	// keep these for smooth animations
	static float dT_display = 20000;
	static float smooth_memptr = 0;
	static Mravenec* last_selected;


	int y = screenHeight - STATS_HEIGHT;
	char c[256];

	dT_display = 0.1f * sim.frameDuration() + 0.9f * dT_display;
	sprintf(c, "SPEED:%3dx   dT:%6dus", sim.ticksPerFrame(), int(dT_display));
	drawString(c, 0, y++);
	sprintf(c, "TICKS:%10d   TIME:%10ds", sim.ticksElapsed(), sim.micros() / 1000000L);
	drawString(c, 0, y++);

	if (justTime) return;

	Level& level = sim.level();
	Tile* t = sim.selectedTile();

	sprintf(c, "CURSOR -FREE- at [%2d,%2d]:%-10s STATE:%5d " BYTE_TO_BINARY_PATTERN " " BYTE_TO_BINARY_PATTERN,
		sim.panX(), sim.panY(), TILE_TYPE_name[int(t->type)], t->state, BYTE_TO_BINARY(t->state >> 8), BYTE_TO_BINARY(t->state));
	drawString(c, 0, y++);
	if (sim.target() != nullptr) drawString("LOCKED", 7, y - 1, FG_GREEN | FG_BRIGHT);
	sprintf_s(c, "TILE: heat=%-10d rng=%-10d", t->temperature, t->random);
	drawString(c, 0, y++);

	Mravenec* u = sim.selectedUnit();
	if (u != nullptr)
	{
		sprintf_s(c, "ANT: ip=%-10d mp=%-10d E=%-10d heat=%-10d growth=%-10d state=" BYTE_TO_BINARY_PATTERN,
			u->irPtr, u->memPtr, u->energy, u->temperature, u->eggGrowth, BYTE_TO_BINARY(u->state));
		drawString(c, 0, y++);

		sprintf_s(c, "PROGRAM:");
		drawString(c, 0, y);
		y += 2;

		// draw the bf program
		for (int x = 0; x < screenWidth; x++)
		{
			int progmemPos = mod(x + u->irPtr - screenWidth / 2, UNIT_PROGMEM_IRCOUNT);
			int progmPos_copy = (progmemPos / 10) * 10;

			if (progmemPos % 10 == 0) chiBuffer[x + (y - 1) * screenWidth].Char.AsciiChar = 179; // |
			for (int i = 0; i < 3; i++)
			{
				if ((progmemPos == 4 || progmPos_copy > 0) && progmemPos % 10 == 4 - i)
				{
					chiBuffer[x + (y - 1) * screenWidth].Char.AsciiChar = (progmPos_copy % 10) + '0';
				}
				progmPos_copy /= 10;
			}

			chiBuffer[x + y * screenWidth].Char.AsciiChar = IR_MNEM[int(u->instructionAt(progmemPos))];
			chiBuffer[x + y * screenWidth].Attributes = progmemPos == u->irPtr ? BG_CYAN | FG_GRAY | FG_BRIGHT : 0x07;
		}
		y++;

		if (last_selected != u)
		{
			smooth_memptr = mod(u->memPtr, UNIT_DYNMEM_SIZE);
		}

		// draw the tape
		for (int x = 0; x < screenWidth; x++)
		{
			int memptr = mod(u->memPtr, UNIT_DYNMEM_SIZE);

			constexpr float q = 0.0002;
			smooth_memptr = (1 - q) * smooth_memptr + q * memptr;


			int mempos = mod(x / 4 - 6 + (int)smooth_memptr, UNIT_DYNMEM_SIZE);
			int offset_x = int((smooth_memptr - (int)smooth_memptr) * 4);
			int irpos = x & 3;

			if (irpos == 0)
			{
				sprintf(c, "%3d", u->dynmemAt(mempos));
				drawString(c, x + 1 - offset_x, y, mempos == memptr ? BG_BRIGHT | BG_YELLOW : mempos % 2 ? FG_GRAY : BG_BRIGHT | FG_WHITE);
			}
		}
	}
	if (u != last_selected || u == nullptr)
	{
		last_selected = u;
	}



	/**
	y += 2;
	sprintf(c, "Tiles mem:%8d B    Units mem:\n%8d B", sim.tilesMemory(), sim.unitsMemory());
	drawString(c, 0, y++);
	y++;

	for (int i = 0; y < screenHeight && i < level.unitCap; i++)
	{
		level.mravenci[i].toString(c, i);
		drawString(c, 1, y++);
	}
	/**/
}
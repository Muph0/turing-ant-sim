#pragma once

#define _AMD64_
#include <Windows.h>
#include <ctime>
#include <cstdint>

#include "sim.h"
#include "utils.h"

extern int screenWidth;
extern int screenHeight;

constexpr uint8_t FG_BLACK = 0;
constexpr uint8_t FG_BLUE = 1;
constexpr uint8_t FG_GREEN = 2;
constexpr uint8_t FG_CYAN = 3;
constexpr uint8_t FG_RED = 4;
constexpr uint8_t FG_MAGENTA = 5;
constexpr uint8_t FG_YELLOW = 6;
constexpr uint8_t FG_GRAY = 7;
constexpr uint8_t FG_BRIGHT = 8;
constexpr uint8_t FG_WHITE = FG_BRIGHT | FG_GRAY;

constexpr uint8_t BG_BLACK = 0 << 4;
constexpr uint8_t BG_BLUE = 1 << 4;
constexpr uint8_t BG_GREEN = 2 << 4;
constexpr uint8_t BG_CYAN = 3 << 4;
constexpr uint8_t BG_RED = 4 << 4;
constexpr uint8_t BG_MAGENTA = 5 << 4;
constexpr uint8_t BG_YELLOW = 6 << 4;
constexpr uint8_t BG_GRAY = 7 << 4;
constexpr uint8_t BG_BRIGHT = 8 << 4;
constexpr uint8_t BG_WHITE = BG_BRIGHT | BG_GRAY;

constexpr uint8_t BRIGHT = FG_BRIGHT | BG_BRIGHT;

//inline unsigned char CharColor(unsigned char hrgb)
//{
//    return hrgb;
//}

void initScreen();

void drawLevel(Simulation &sim);
void drawString(char* str, int posx, int posy, WORD attributes);
void drawStats(Simulation& sim, bool justTime = false);
void clearAndResizeBuffer(WORD attributes = FG_GRAY);
void drawHotkeys(Simulation& sim);
void drawFlush();
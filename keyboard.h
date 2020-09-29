#pragma once

#include <cstdint>
#define _AMD64_
#include <Windows.h>

void keyboardUpdate();

bool isKeyDown(int vKey);
bool isKeyPressed(int vKey);
bool isKeyReleased(int vKey);


#include "keyboard.h"

#define KEYMAP_SIZE 512
#define KEY_TOGGLED   1
#define KEY_DOWN    128

uint8_t oldKeyboardState[KEYMAP_SIZE];
uint8_t keyboardState[KEYMAP_SIZE];

bool keyboardInitialised = false;

void keyboardUpdate()
{
    if (!keyboardInitialised)
    {
        keyboardInitialised = true;

        for (int i = 0; i < KEYMAP_SIZE; i++)
        {
            oldKeyboardState[i] = false;
            keyboardState[i] = false;
        }
    }

    HWND consolew = GetConsoleWindow();
    HWND activew = GetForegroundWindow();

    for (int i = 0; i < KEYMAP_SIZE; i++)
    {
        oldKeyboardState[i] = keyboardState[i];
        if (consolew == activew)
            keyboardState[i] = GetKeyState(i);
    }
}

#define KEY_TOGGLED   1
#define KEY_DOWN    128

bool isKeyDown(int vKey)
{
    return (keyboardState[vKey] & KEY_DOWN) != 0;
}
bool isKeyPressed(int vKey)
{
    return ((keyboardState[vKey] & KEY_DOWN) != 0) && ((oldKeyboardState[vKey] & KEY_DOWN) == 0);
}
bool isKeyReleased(int vKey)
{
    return ((keyboardState[vKey] & KEY_DOWN) == 0) && ((oldKeyboardState[vKey] & KEY_DOWN) != 0);
}
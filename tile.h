#pragma once

enum class TileType : uint8_t
{
    Void = 0,
    Mravenec = 1,
    Wall = 2,
    Food = 3,
    Egg = 4,
    Count
};

#define TileType (uint8_t)TileType

constexpr char* TILE_TYPE_name[TileType::Count] = { "VOID", "UNIT", "WALL", "FOOD", "EGG" };

struct Tile {
    uint8_t type;
    uint8_t temperature;
    uint16_t state;
    uint32_t random;
};

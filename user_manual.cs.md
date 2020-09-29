# Uživatelská příručka

Program turing-ant-sim slouží k simulaci života turingových mravenců. (viz [zadání](https://github.com/Muph0/turing-ant-sim/blob/master/README.cs.md))

## Spuštění

Program vyžaduje jeden parametr z příkazové řádky, a to je název souboru, ze kterého se má stav simulace načíst
a do kterého se budou budoucí stavy ukládat. Pokud tento soubor neexistuje, bude vytvořen a zapíše se do něj
počáteční stav

## Ovládání

Program se ovládá pomocí klávesnice.

|Klávesa|                   |
|:-----:|-------------------|
| Šipky | Posouvání kurzoru v simulaci. 
| Enter | Jeden krok simulace
|  +/-  | Změna rychlosti automatického běhu simulace
|   A   | Rychlejší posouvání. 
|   W   | Stav simulace se vypíše do souboru označeného počtem uběhnutých iterací 
|   R   | Znovu se načtou [konfigurační parametry](#konfigurační-parametry)
|   L   | Pozice kurzoru se zamkne na aktuálně vybraného mravence
|   F   | Přepne zobrazení do režimu feromonů
|   G   | Zobrazí poslední cifru stavu generátoru pseudonáhody v každé dlaždici
|   H   | Zobrazí rozložení tepla v simulaci

## Zobrazení

![zobrazení](https://i.imgur.com/WaDErrD.png)

V horní části obrazovky je pole, které představuje vizualizaci dlaždic simulace. Červené dlaždice s šipkou jsou mravenci,
žlutá jsou vajíčka, a zelenou barvou je znázorněna potrava. Bílá dlaždice je zvýrazněný mravenec, protože se na něm nachází kurzor.

Ve spodní části jsou informace o simulaci, popořadě:
- `SPEED` - Rychlost simulace. Udává kolik kroků proběhne během jednoho snímku.
- `dT` - Délka jednoho snímku v mikrosekundách.
- `TICKS` - Počet kroků, které proběhly od spuštění simulace.
- `TIME` - Doba běhu aktuální instance.
- `CURSOR` - Informace o zamčení kurzoru a jeho souřádnice. Za dvoujtečkou je typ dlaždice pod kurzorem.
- `STATE:` - hodnota `state` aktuální dlaždice.
- `TILE:` - ostatní informace o dlaždici.
- `ANT:` - informace o stavu mravence
  - `ip` je ukazatel na aktuálně vykonávanou instrukci.
  - `mp` je ukazatel na pásku.
  - `E` je energie mravence
  - `heat` je teplota mravence
  - `growth` je růst mravence, má význam jen pro vajíčka
  - `state` jsou různé příznaky které si mravenec nese. Spodní dva bity jsou jeho směr, třetí bit je 1 pro živé mravence,
    a další tři bity jsou aktuální rozhodnutí mravence.
- `PROGRAM:` - stupnice s čísly označuje každou desátou instrukci, a pod ní je okno do instrukční paměti mravence. (viz [formát kódu mravence](#formát-kódu-mravence))
  Aktuální instrukce je zvýrazněná uprostřed obrazovky. Pod tím je okno do pracovní paměti mravence, žlutě je zvýrazněna aktuální pozice.

## Ukládání stavu

Soubor se stavem mravence obsahuje 4 parametry a 3 bloky

Parametry:

```plain
width:      - šířka simulace v dlaždicích
height:     - výška simulace v dlaždicích
max_units:  - maximální počet mravenců v simulaci
ticks:      - počet kroků které uběhly od vytvoření simulace
```

Bloky:

```plain
params    { ... }   - blok s konfiguračními parametry
unit_data { ... }   - blok s informacemi o mravencích
tile_data { ... }   - blok s informacemi o dlaždicích
```

### Konfigurační parametry

V bloku `params` jsou konfigurační parametry simulace, které lze měnit za průběhu

```plain
params {
    food_growth_speed:      - pravděpodobnost, že na dlaždici povyroste jídlo
    food_growth_step:       - o kolik jídlo povyroste
    food_growth_max:        - maximální energie kterou jídlo může mít
    movement_cost:          - množství energie, které mravenec spálí na 1 pohyb
    heating_amount:         - množství tepla, které mravenec vyrobí z 1 energie
    energy_death_th:        - práh energie, pod který když se energie mravence dostane, tak mravenec zemře
    temp_death_th:          - práh teploty, pod který když se teplota mravence dostane, tak mravenec zemře
    ambient_temp:           - okolní teplota, dlaždice sousedící s okrajem sousedí s virtuálními dlaždicemi o této teplotě
    egg_laying_th:          - kolik energie musí mravenec investovat do vajíčka, aby se provedlo jeho položení
    egg_growth:             - pravděpodobnost, že v 1 kroku vajíčko povyroste o 1
    ground_conductivity:    - tepelná vodivost podlahy
    wall_conductivity:      - tepelná vodivost překážky
    ant_conductivity:       - tepelná vodivost mravence
}
```

### Formát kódu mravence

Kód mravence je nadmnožinou jazyka Brain*uck. Každý znak představuje jednu instrukci.

| IR | Význam
|:--:|-------
| >  | `memPtr++`
| <  | `memPtr–-`
| +  | `dynmem[memPtr]++`
| -  | `dynmem[memPtr]--`
| [  | `while (dynmem[memPtr] > 0) {`
| ]  | `}`
| &  | `dynmem[memPtr] += dynmem[memPtr - 1]`
| =  | `dynmem[memPtr] -= dynmem[memPtr - 1]`
| *  | `dynmem[memPtr] <<= 1`
| /  | `dynmem[memPtr] >>= 1`
| ! [<sup>(2)</sup>](vajicko) | Pohyb mravence. Podle `dynmem[memPtr]` se posune vpřed, otočí doprava nebo doleva, a úspěch se zapíše do `dynmem[memPtr]`.
| ? [<sup>(2)</sup>](vajicko) | Do parametru se zapíše *feromonová* hodnota pole před mravencem.
| $ [<sup>(2)</sup>](vajicko) | Mravenec se pokusí sníst, co je na dlaždici před ním. Pokud je tam jiný mravenec nebo vajíčko, sebere mu část energie úměrnou jeho energii.
| . [<sup>(2)</sup>](vajicko) | Mravenec před sebe položí vajíčko, a předá mu `dynmem[memPtr - 1]` energie. Pokud to je méně než `egg_laying_th`, tak položí překážku.
| %  | Mravenec použije 1 energie na výrobu tepla.
| @  | Do `dynmem[memPtr]` se zapíše teplota v mravence.

<a name="vajicko">**(2):**</a> Nefunguje ve stádiu vajíčka.



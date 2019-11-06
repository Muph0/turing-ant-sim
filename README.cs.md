Simulátor evoluce chování turingových mravenců
==============================================

[<sup>(1)</sup>]: #proudeni "Proč ne proudění?"
[<sup>(2)</sup>]: #vajicko "Nefunguje ve stádiu vajíčka."


### Zadání programu

**Turingův mravenec** - to je mravenec mající místo mozku malý turingův stroj, který vykonává
nějaký program a tak říká mravenci, co má dělat. Mravenec se pohybuje v dvourozměrné mřížce
a to ho stojí energii. Takže si musí hledat potravu aby si ji doplnil. Nese si svou genetickou
informaci (to je jeho program), a tu, pokud má dostatek energie, může předat část svým kolegům
mravencům nebo svým potomkům - při čemž dochází k náhodným mutacím v kódu (AAAAA -> AAAXA),
přerovnání úseků (ABCDE -> CDABE)  nebo k přesmyku v případě pohlavního rozmnožování
(rodič 1: AAAAA + rodič 2: BBBBB -> AAABB).

Aby toho nebylo málo, venku je zima, proto se mravencům vyplatí stavět mraveniště, ve kterých
se mohou shlukovat a vyrábět si teplo, které jim umožní se efektivněji pohybovat, a hlavně se
efektivněji rozmnožovat - položit vajíčko stojí mravence energii, kterou vajíčku předá. Vajíčko
ji může použít pro zahřátí aby neuhynulo - to by mělo mravencům co staví mraveniště dát evoluční
výhodu. Teplota se šíří pouze vedením[<sup>(1)</sup>], a zdi mraveniště vodí dost pomalu
na to, aby se mravenci mohli izolovat.

Úkolem mého programu je vygenerovat náhodnou populaci mravenců (příp. načíst už existující ze
souboru) a vytvořit pro tyto mravence prostředí kde mohou existovat. Odsimulovat několik tisíc
generací mravenců (ideálně co nejvíce), do sytosti uživatele, během toho vše vizualizovat a
případně zaznamenat nějaké statistické informace, a nakonec všechny mravence opět uložit do
souboru pro další případné simulace.

Přes prázdniny jsem udělal malou zkušenost s nvidia CUDA, a díky probrané látce za říjen si
myslím, že bych mohl v c++ napsat program co by simuloval tyto mravence paralelně, tj. dost
rychle na to rychle, aby se se dala pozorovat evoluce v jejich chování.

<br>
<br>

**********

<a name="proudeni">**(1):**</a> Proudění by bylo fajn, ale myslím že automat na simulování proudění vzduchu
o různých teplotách je už samo o sobě téma pro zápočtovou práci.


**********

### Detaily zadání

*Poznámka: náseldující detaily jsou pouze předběžné, je možné že se cokoliv ještě změní,
třeba kvůli optimalizaci, nebo kvůli zásahu cvičícího.*

#### Parametry simulace

    food_growth_prob          - pravděpodobnost, se kterou na dlaždici vyroste jídlo
    movement_cost             - množství energie, kterou mravenec zaplatí za 1 pohyb
    heating_amount            - o kolik mraveneci stoupne teplota z 1 energie
    energy_death_th           - práh energie, který mravenec nesmí překročit aby přežil
    heat_death_th             - práh tělesného tepla, který mravenec nesmí překročit aby přežil
    ambient_heat              - množství tepla, které má dlaždice po tom co na ní vyroste jídlo
    egg_laying_th             - minimální množství energie, kterou musí mravenec dát vajíčku
    egg_growth_prob           - pravděpodobnost, se kterou vajíčko vyroste o 1

#### Život mravence

Aby mravenec mohl žít, musí dodržovat následující pravidla:

  1. Jeho energie musí být > `energy_death_threshold`
  2. Jeho tělesná teplota musí být > `heat_death_threshold`

Mravenec může přeměňovat energii na teplo, ale to ztrácí výměnou s prostředím, proto musí buď být
v teple, nebo neustále hledat potravu.

Aby se planeta mravenců časem nepřehřála, vždy když někde vyroste nové jídlo, tak je hodnota teploty
této a okolních dlaždic nastavena na `ambient_heat`.

#### Roznmožování mravenců

Rozmožování probíhá takto

  1. Mravenec nahromadí dostatečné množství energie
  2. Položí před sebe vajíčko a část své energie mu předá. (v tuto chvíli vzniká nový jedinec)
  3. Tímto končí nepohlavní rozmnožování, ale než se vajíčko vylíhne,
     tak mají ostatní mravenci možnost do něj vkládat svůj gen tak, že 
     překryjí část kódu vajíčka svým kódem.
     
Mravenec ve stádiu vajíčka vykonává svůj kód normálně, jen všechny jeho funkce pro interakci s okolím nemají žádný účinek,
a vrací mravenci negativní výsledky.

Mravenci se tedy mohou rozmnožovat pohlavně i nepohlavně - tj. potomek má genetickou informaci pouze od jednoho rodiče
nebo od více rodičů.

**********

### Detaily implementace

#### Mřížka
  
Dlaždice jsou rozděleny do čtvercových bloků velkých 32x32. Nad každým běží 32x32 CUDA vláken, pro každou dlaždici jedno.
Mravenci leží v [globální paměti](https://www.3dgep.com/cuda-memory-model/#Global), která je sice pomalá, ale
sdílená paměť u mojí grafické karty je příliš malá na to aby se do ní vešlo až 512 mravenců, a kdybych zmenšil
bloky, zase by se zvýšil počet situací kdy mravenec musí přejít z jednoho bloku do druhého, a kopírování celého
mravence (včetně jeho programu) by zabralo ještě více času. Takhle se přenáší vždy pouze jedna instrukce a jiné
malé parametry mravence.

Každá dlaždice si nese svůj typ, stav pseudonáhodného generátoru, teplotu, a feromonovou hodnotu, tj číslo `dynmem[0] & 0x7F`
posledního mravence co na ní stál, tedy pokud na ní zrovna mravenec nestojí - pak je to ukazatel přímo na toho mravence.
Typ dlaždice je buď `VZDUCH`, `MRAVENEC`, `JÍDLO`, `PŘEKÁŽKA` nebo `VAJÍČKO`.

#### Mravencův program

Mravenec interpretuje verzi [jazyka P''](https://en.wikipedia.org/wiki/P%E2%80%B2%E2%80%B2) rozšířenou o instrukce pro základní aritmetiku a pro interakci se svým prostředím.

V paměti vypadá nějak takhle:

    struct Mravenec {
        int posx, posy;
        int direction;
        bool alive;
        int age;
        float energy, heat;
        int irPtr, memPtr;
        uint8_t progmem[PROGMEM_SIZE];
        int8_t dynmem[DYNMEM_SIZE];
    }
    
Pokud je nějaké číslo interpretované jako směr, například pole `direction`, znamená to

    case (direction & 3)
      0: positive x
      1: positive y
      2: negative x
      3: negative y

`alive` je vlajka alive pro vlákno, které uklízí mrtvoly mravenců v daném bloku.
Instrukcí je 15 druhů, tedy se do každého bajtu vejdou dvě.

V následující tabulce
`dynmem[memPtr - 1]` ve skutečnosti znamená `dynmem[mod(memPtr - 1, DYNMEM_SIZE)]`.

| IR | Význam
|:--:|-------
| >  | `memPtr++;`
| <  | `memPtr–-;`
| +  | `dynmem[memPtr]++;`
| -  | `dynmem[memPtr]--;`
| [  | `while (dynmem[memPtr] > 0) {`
| ]  | `}`
| #  | `dynmem[memPtr] += ;`
| =  | `dynmem[memPtr] -= dynmem[mod(memPtr - 1, DYNMEM_SIZE)];`
| ^  | `dynmem[memPtr] <<= 1;`
| v  | `dynmem[memPtr] >>= 1;`
| ! [<sup>(2)</sup>] | Pohyb mravence. Podle `dynmem[memPtr - 1]` se posune vpřed, otočí doprava nebo doleva, a úspěch se zapíše do `dynmem[memPtr]`.
| ? [<sup>(2)</sup>] | Do parametru se zapíše *feromonová* hodnota pole před mravencem, navíc | 0x80 pokud je tam překážka, vajíčko nebo mravenec.
| @ [<sup>(2)</sup>] | Mravenec se pokusí sníst, co je na dlaždici před ním.
| % [<sup>(2)</sup>] | Mravenec před sebe položí vajíčko, a předá mu `dynmem[memPtr - 1]` energie. Pokud to je méně než `egg_laying_th`, tak položí překážku.
| *  | Mravenec použije `dynmem[memPtr - 1]` energie na výrobu tepla, do `dynmem[memPtr]` se zapíše hodnota teploty po zahřátí.
| &  | Do `dynmem[memPtr]` se zapíše teplota v okolí mravence.

<a name="vajicko">**(2):**</a> Nefunguje ve stádiu vajíčka.

#### Náhoda

Pro generování pseudonáhody používám 32-bit linear-feedback shift register, v každé dlaždici jeden, a protože 1 dlaždice = 1 vlákno,
každé vlákno má svojí vlastní náhodu, takže se nemusí prát o nějakou společnou paměť.








# seedwright

Clean-room Minecraft (vanilla, Java Edition) seed generation + seed finder, CPU and CUDA, versions
**1.21.1 – 26.3**. Backend + CLI; no rendering, no lighting, no chunk generation beyond what the filters need.

Written from vanilla's own (deobfuscated) worldgen sources and data packs, checked against the real game.
No third-party seed tools were read.

## What is bit-exact (verified against vanilla, all 16 versions)

`tools/verify-all.sh` rebuilds every version pack and compares with ground truth dumped by the headless
oracle mod (real dedicated server, 2 seeds per version):

| part | check |
|---|---|
| climate noise (temperature, vegetation, continentalness, erosion, depth, weirdness) | 4000 random points/seed, float bits identical |
| biomes (point path) | 4000 points/seed, identical |
| biomes (26.3 column/volume path, used by the structure pre-check) | 19 400 quarts/seed, identical |
| spawn origin chunk (NoiseSpawnFinder) | identical |
| strongholds (first ring sets, 12 positions) | identical |
| structure placement candidates (spread, frequency reducers, exclusion zones) | all sets within 40 chunks, identical |

**Approximate:** whether a structure actually starts (its biome check uses the terrain height there).
Seedwright uses the depth=0 surface instead. Result on the truth seeds: 126/127 starts match (84/84 and
42–43/43–46: a few mineshafts/ruined portals differ). Exact terrain height is the next step.

Version differences handled: 26.3's new float noise/density engine vs. the double-precision legacy engine
(1.21.1–26.2), RTree fan-out (6 before 26.3, 19 since), the 1.21.1 spawn-fitness formula, and data-pack
differences (pale garden, abandoned camps, …) via per-version packs.

## Build

```
cd engine && make            # seedwright-cpu (g++/OpenMP) and seedwright-gpu (nvcc, sm_120)
```
Bit-exactness needs `-ffp-contract=off` / `-fmad=false` (already in the Makefile).

## Use

```
# a plains village with 2 blacksmiths within 500 blocks of spawn, a cherry grove within 1500,
# and no ocean monument within 800
engine/seedwright-gpu packs/26.3.pack search --where \
  '2 village_plains within 500 { 2 blacksmith }, biome cherry_grove within 1500, not monument within 800'

# lush caves under spawn, and either a trial chamber or an ancient city nearby
engine/seedwright-gpu packs/26.3.pack search --where \
  'biome lush_caves within 100 at y=any, any { trial_chambers within 600, ancient_city within 800 }'

# check given seeds (always exact)
engine/seedwright-cpu packs/1.21.4.pack search --cpu --where 'trial_chambers within 400' --seeds 12345,8675309
```

### Query language (`--where`)

Conditions are separated by `,` (or `and`); everything is measured from the world spawn (`--origin`: from 0,0).

| form | meaning |
|---|---|
| `[N] STRUCTURE within R [{ ... }]` | at least N (default 1) within R blocks; inside `{ }` is measured from that structure |
| `biome NAME within R [at y=Y \| at y=any]` | biome on a 16-block grid; default y=64; `y=any` tries y -48..48 (cave biomes: `lush_caves`, `dripstone_caves`, `deep_dark`) |
| `[N] strongholds within R [{ ... }]` | strongholds from any ring (up to 32) |
| `[N] BUILDING` | inside a village's `{ }`: buildings in its layout, e.g. `2 blacksmith`, `library`, `temple` (approximate, see below) |
| `spawn within R` | the world spawn itself is within R blocks of 0,0 (only useful for "spawn near the map centre") |
| `any { a, b }`, `all { a, b }`, `not X`, `a or b`, `( )` | logic |

STRUCTURE is a set (`villages`, `village`), an exact structure (`village_plains`), or a `*` pattern
(`village_*`, `*ruin*`). Buildings accept vanilla's per-biome names and job names: `blacksmith`/`weaponsmith`,
`toolsmith`, `armorer`, `butcher`, `mason`, `shepherd`, `fisher`, `fletcher`, `cartographer`, `library`/`librarian`,
`temple`/`cleric`, `tannery`/`leatherworker`, `small_house`, `large_farm`, … (`seedwright <pack> list` prints all
names; `seedwright <pack> parse '<query>'` prints the parsed tree as JSON). Old flags still work and can be mixed in:
`--need NAME:COUNT@RADIUS` (substring match, `=NAME` exact), `--biome NAME@RADIUS[@Y|@any]`, `--stronghold COUNT@RADIUS`,
`--spawn-within RADIUS`.

Options: `--start/--count` (low-48 range), `--max N`, `--slack B`, `--cpu`, `--seeds a,b,c`.
`--slack` (default 256) is a speed shortcut for searches: seeds whose spawn is farther than that from 0,0 are skipped
(about 2% of seeds at 256). `--seeds` checks never use it.

**Village buildings are approximate.** Layout assembly (jigsaw) matches vanilla piece for piece when given vanilla's
terrain heights (149/149 villages and outposts against the oracle), but seedwright still estimates terrain height,
and layouts are chaotic in it: per-category building counts agree with the game ~66% of the time. Use building
conditions to find candidates, then check in game. Exact terrain height makes them exact without query changes.

GUI (PySide6): `python3 gui/seedwright_gui.py`. Scratch-style blocks for the same language: drag blocks from the
palette into the script, drop blocks into C-blocks (structures, strongholds, all/any/none) to measure them from
there, drag a block back onto the palette to delete it. The query text under the blocks is the `--where` string and
can be edited and turned back into blocks. Menu launcher: `gui/install.sh` (writes
`~/.local/share/applications/seedwright-gui.desktop` pointing at this checkout; `doas gui/install.sh --system`
puts it in `/usr/share/applications` for launchers that only read that, like tux; `--uninstall` removes it).

Pipeline: stage A tests only the low 48 bits (structure placement is pure LCG) at ~5.7 G seeds/s on an
RTX 5070 Ti; stage B expands each survivor's 65 536 upper halves with full biome/spawn checks
(~185 k seeds/s on 26.3, ~60 k on legacy versions; CPU about 1/3 of that on 16 threads).

## Layout

- `engine/sw_core.h`: RNGs, both noise engines, density-function program, RTree, spawn, placement (host+device)
- `engine/structures.h`: structure validation, strongholds
- `engine/query.h`, `engine/query_parse.h`: the condition tree (stage A / prefilter / exact evaluation) and the query language
- `engine/jigsaw.h`: village (jigsaw) layouts
- `engine/pack.h`: version-pack loader, RTree build
- `tools/extract.py`: vanilla jar (+ oracle biome list) → `packs/<version>.pack`
- `oracle/` (26.x, unobfuscated Loom) and `oracle-legacy/` (1.21.x, remapping Loom): headless ground-truth mod;
  `oracle/tools/truth.sh` / `all-truth.sh`; output in `truth/<version>/`
- `tools/verify-all.sh`: rebuild + verify everything

## Not done yet

- exact terrain height (final density) → exact structure starts, exact village buildings, real spawn block, surface biomes
- ores (decoration seeds + ore placement + whether the blocks survive terrain/caves)
- `--biome` at the real surface; nether/end
- GPU stage B is one thread per seed; a block-per-seed version would be much faster
- stronghold conditions are slow (~1–2k seeds/s on GPU: each stronghold near the edge of the radius needs a
  57×57 biome search); jigsaw structures other than villages/outposts aren't oracle-checked; trial chambers have no layouts

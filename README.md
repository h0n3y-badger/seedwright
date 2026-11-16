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
# 3 villages within 300 blocks of spawn, a monument within 800, a cherry grove within 1500
engine/seedwright-gpu packs/26.3.pack search --need village:3@300 --need monument:1@800 --biome cherry_grove@1500

# check specific seeds
engine/seedwright-cpu packs/1.21.4.pack search --cpu --need trial_chambers:1@400 --seeds 12345,8675309
```
Filters: `--need NAME:COUNT@RADIUS` (structure or set name, substring ok: `village`, `village_snowy`,
`monument`, `trial_chambers`, `outpost`, `abandoned_camp`, …), `--biome NAME@RADIUS` (y=64, 16-block grid),
`--stronghold COUNT@RADIUS` (first ring), `--spawn-within RADIUS`. Distances are measured from the spawn origin
(`--origin` for 0,0). Options: `--start/--count` (low-48 range), `--max N`, `--slack B` (how far spawn may
be from 0,0 for the structure pre-filter, default 256), `--cpu`.

Pipeline: stage A tests only the low 48 bits (structure placement is pure LCG) at ~5.7 G seeds/s on an
RTX 5070 Ti; stage B expands each survivor's 65 536 upper halves with full biome/spawn checks
(~185 k seeds/s on 26.3, ~60 k on legacy versions; CPU about 1/3 of that on 16 threads).

## Layout

- `engine/sw_core.h`: RNGs, both noise engines, density-function program, RTree, spawn, placement (host+device)
- `engine/structures.h`, `engine/query.h`: structure validation, strongholds, filters
- `engine/pack.h`: version-pack loader, RTree build
- `tools/extract.py`: vanilla jar (+ oracle biome list) → `packs/<version>.pack`
- `oracle/` (26.x, unobfuscated Loom) and `oracle-legacy/` (1.21.x, remapping Loom): headless ground-truth mod;
  `oracle/tools/truth.sh` / `all-truth.sh`; output in `truth/<version>/`
- `tools/verify-all.sh`: rebuild + verify everything

## Not done yet

- exact terrain height (final density) → exact structure starts, real spawn block, surface biomes
- ores (decoration seeds + ore placement + whether the blocks survive terrain/caves)
- `--biome` at the real surface; nether/end
- GPU stage B is one thread per seed; a block-per-seed version would be much faster

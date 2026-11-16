// Seed cracking (host): recover world seeds from things seen in a world.
//   seedwright <pack> crack STRUCTURE@X,Z ... [spawn@X,Z[~TOL]] [--max N]
// X,Z are block coordinates inside the structure's start chunk (what /locate prints).
//
// 1. Low 48 bits (the "structure seed"): random_spread placement is a pure function of the low 48 bits.
//    nextInt(n) with n = 2^k * odd (k > 0) gives (next(31) mod 2^k) = bits 17..17+k-1 of the LCG state, and the
//    low b bits of the state depend only on the low b bits of the seed. So each linear placement offset pins
//    k bits of the low 17+k bits: enumerate those first (2^20 at most), then the remaining high bits for the
//    survivors, checking every clue's placement exactly (isStructureChunk: offsets, frequency, exclusion).
// 2. Upper 16 bits: biome-dependent. For each 48-bit candidate try all 65536 and keep the seeds where every
//    clue's chunk really generates that structure (the start's biome check) and the spawn origin is near spawn@.
// nextInt's rejection loop (probability ~ n / 2^31 per call) is ignored in step 1; such a clue just fails.
#pragma once
#include "structures.h"
#include "jigsaw.h"
#include <atomic>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>
#include <omp.h>

namespace sw {

struct CrackClue {
	int st, set;   // structure index, its set
	int cx, cz;    // start chunk
	int rx, rz;    // region
	int ox, oz;    // offsets in the region
};

static const uint64_t LCG_M = 0x5DEECE66DULL, LCG_A = 0xBULL, MASK48 = (1ULL << 48) - 1;

// does seed's low `bits` bits agree with clue c (only the 2^k part of each offset)?
static inline bool crackLowOk(const StructTables& T, const CrackClue& c, uint64_t low, int bits) {
	const Spread& s = T.sets[c.set].spread;
	int lim = s.spacing - s.separation;
	int k = __builtin_ctz(lim);
	if (s.triangular || (lim & (lim - 1)) == 0 || 17 + k > bits) return true;
	uint64_t m = (1ULL << (17 + k)) - 1;
	uint64_t st = (((uint64_t)((int64_t)c.rx * 341873128712LL + (int64_t)c.rz * 132897987541LL) + low + (uint64_t)(int64_t)s.salt) ^ LCG_M) & m;
	uint64_t km = (1ULL << k) - 1;
	st = (st * LCG_M + LCG_A) & m;
	if (((st >> 17) & km) != ((uint64_t)c.ox & km)) return false;
	st = (st * LCG_M + LCG_A) & m;
	return ((st >> 17) & km) == ((uint64_t)c.oz & km);
}

static int crackCommand(const Pack& P, const StructTables& T, const JigsawTables* J, int argc, char** argv) {
	std::vector<CrackClue> clues;
	bool haveSpawn = false;
	int spX = 0, spZ = 0, spTol = 128;
	long long maxOut = 50, maxStructSeeds = 64;
	for (int i = 3; i < argc; i++) {
		std::string a = argv[i];
		if (a == "--max" && i + 1 < argc) {
			maxOut = atoll(argv[++i]);
			continue;
		}
		if (a == "--max-structure-seeds" && i + 1 < argc) {
			maxStructSeeds = atoll(argv[++i]);
			continue;
		}
		size_t at = a.find('@'), comma = a.find(',', at);
		if (at == std::string::npos || comma == std::string::npos) {
			fprintf(stderr, "bad clue '%s' (want NAME@X,Z)\n", a.c_str());
			return 2;
		}
		std::string name = a.substr(0, at);
		int x = std::stoi(a.substr(at + 1, comma - at - 1));
		size_t tilde = a.find('~', comma);
		int z = std::stoi(a.substr(comma + 1, tilde == std::string::npos ? std::string::npos : tilde - comma - 1));
		if (name == "spawn") {
			haveSpawn = true, spX = x, spZ = z;
			if (tilde != std::string::npos) spTol = std::stoi(a.substr(tilde + 1));
			continue;
		}
		CrackClue c{};
		c.st = -1;
		for (size_t s = 0; s < P.structures.size(); s++)
			if (P.structures[s].name.substr(P.structures[s].name.find(':') + 1) == name) c.st = (int)s;
		if (c.st < 0) {
			fprintf(stderr, "unknown structure '%s' (see `list`)\n", name.c_str());
			return 2;
		}
		c.set = -1;
		for (size_t s = 0; s < P.sets.size() && c.set < 0; s++)
			for (int x2 : P.sets[s].structs)
				if (x2 == c.st && T.sets[s].type == SET_SPREAD) c.set = (int)s;
		if (c.set < 0) {
			fprintf(stderr, "'%s' isn't placed on a random spread grid; it can't be used as a clue\n", name.c_str());
			return 2;
		}
		const Spread& sp = T.sets[c.set].spread;
		c.cx = floorDiv(x, 16), c.cz = floorDiv(z, 16);
		c.rx = floorDiv(c.cx, sp.spacing), c.rz = floorDiv(c.cz, sp.spacing);
		c.ox = c.cx - c.rx * sp.spacing, c.oz = c.cz - c.rz * sp.spacing;
		if (c.ox >= sp.spacing - sp.separation || c.oz >= sp.spacing - sp.separation) {
			fprintf(stderr, "%s@%d,%d: no %s can start in chunk %d,%d (outside its placement area); is it the start chunk?\n", name.c_str(), x, z,
			        name.c_str(), c.cx, c.cz);
			return 2;
		}
		clues.push_back(c);
	}
	if (clues.empty()) {
		fprintf(stderr, "usage: seedwright <pack> crack STRUCTURE@X,Z ... [spawn@X,Z[~TOL]] [--max N]\n");
		return 2;
	}
	// most selective clues first (fewest candidate chunks per region)
	std::sort(clues.begin(), clues.end(), [&](const CrackClue& a, const CrackClue& b) {
		const Spread &sa = T.sets[a.set].spread, &sb = T.sets[b.set].spread;
		return (sa.spacing - sa.separation) > (sb.spacing - sb.separation);
	});
	double info = 0;
	int lowBits = 17;
	for (auto& c : clues) {
		const Spread& s = T.sets[c.set].spread;
		int lim = s.spacing - s.separation;
		info += 2 * std::log2((double)lim);
		int k = __builtin_ctz(lim);
		if (!s.triangular && (lim & (lim - 1)) != 0) lowBits = std::max(lowBits, 17 + k);
	}
	fprintf(stderr, "%zu clues, ~%.0f bits of placement information (48 needed for a unique structure seed)\n", clues.size(), info);
	auto t0 = std::chrono::steady_clock::now();
	auto secs = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };

	// step 1a: low bits
	std::vector<uint64_t> lows;
	for (uint64_t low = 0; low < (1ULL << lowBits); low++) {
		bool ok = true;
		for (auto& c : clues)
			if (!(ok = crackLowOk(T, c, low, lowBits))) break;
		if (ok) lows.push_back(low);
	}
	fprintf(stderr, "low %d bits: %zu of %llu candidates left\n", lowBits, lows.size(), 1ULL << lowBits);
	// step 1b: high bits of the structure seed, exact placement check
	std::vector<uint64_t> seeds48;
	std::mutex mu;
	std::atomic<long long> done{0};
	const uint64_t nHigh = 1ULL << (48 - lowBits);
#pragma omp parallel for schedule(dynamic, 1)
	for (long long li = 0; li < (long long)lows.size() * 64; li++) {  // 64 slices per low value for load balance
		if ((long long)seeds48.size() > maxStructSeeds) continue;  // already ambiguous: stop early
		uint64_t low = lows[li / 64], h0 = nHigh / 64 * (li % 64), h1 = h0 + nHigh / 64;
		for (uint64_t h = h0; h < h1; h++) {
			uint64_t s = (h << lowBits) | low;
			bool ok = true;
			for (auto& c : clues) {
				int px, pz;
				spreadCandidate(T.sets[c.set].spread, s, c.rx, c.rz, px, pz);
				if (px != c.cx || pz != c.cz) {
					ok = false;
					break;
				}
			}
			if (ok)
				for (auto& c : clues)
					if (!(ok = isStructureChunk(T, c.set, s, c.cx, c.cz))) break;
			if (ok) {
				std::lock_guard<std::mutex> g(mu);
				seeds48.push_back(s);
			}
		}
		long long d = ++done;
		if (omp_get_thread_num() == 0)
			fprintf(stderr, "\rstructure seeds: %.0f%% (%zu found, %.0f s)   ", 100.0 * d / (lows.size() * 64), seeds48.size(), secs());
	}
	std::sort(seeds48.begin(), seeds48.end());
	fprintf(stderr, "\rstructure seeds (low 48 bits): %zu found in %.1f s                \n", seeds48.size(), secs());
	if (seeds48.empty()) {
		fprintf(stderr, "no seed places every clue there: check the coordinates (each must be in the structure's start chunk, e.g. from /locate) and the version\n");
		return 1;
	}
	if ((long long)seeds48.size() > maxStructSeeds) {  // each costs ~1 s of biome checks
		fprintf(stderr, "ambiguous: over %lld structure seeds fit; add more structures (each adds ~9 bits) or raise --max-structure-seeds\n", maxStructSeeds);
		return 3;
	}
	// step 2: upper 16 bits from what generates there (biomes) and the spawn
	long long found = 0;
	const PackDev& pk = *P.dev;
	for (uint64_t s48 : seeds48) {
		std::vector<int64_t> full;
#pragma omp parallel
		{
			World* w = new World();
#pragma omp for schedule(dynamic, 64)
			for (int hi = 0; hi < 65536; hi++) {
				int64_t seed = (int64_t)(((uint64_t)hi << 48) | s48);
				worldInit(*w, pk, seed);
				bool ok = true;
				for (auto& c : clues) {
					int st = J ? jigsawStructureAt(*w, pk, T, *J, c.set, c.cx, c.cz) : structureAt(*w, pk, T, c.set, c.cx, c.cz);
					if (!(ok = st == c.st)) break;
				}
				if (ok && haveSpawn) {
					int sx, sz;
					spawnOrigin(*w, pk, sx, sz);
					ok = std::abs(sx - spX) <= spTol && std::abs(sz - spZ) <= spTol;
				}
				if (ok) {
					std::lock_guard<std::mutex> g(mu);
					full.push_back(seed);
				}
			}
			delete w;
		}
		std::sort(full.begin(), full.end());
		for (int64_t s : full)
			if (found++ < maxOut) printf("%lld\n", (long long)s);
		fflush(stdout);
	}
	fprintf(stderr, "done: %lld world seed(s) in %.1f s%s\n", found, secs(),
	        found > 1 ? " (several fit: more clues or a spawn@ narrows it down)" : found ? "" : " (biomes/spawn ruled out every candidate: check structure types, spawn tolerance, version)");
	return found ? 0 : 1;
}

}  // namespace sw

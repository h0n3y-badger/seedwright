// Seed filters ("x villages within r blocks of spawn", biomes, strongholds...). Device-capable.
#pragma once
#include "structures.h"

namespace sw {

enum CondKind { COND_STRUCTURE, COND_BIOME, COND_STRONGHOLD, COND_SPAWN };

struct Cond {
	int kind, count, radius;
	uint64_t structMask;  // structure indices (bit i = structure i)
	uint32_t setMask;     // sets that can produce one of those structures
	int biome;
};

enum { MAX_CONDS = 12, MAX_HITS = 16 };

struct Query {
	int ncond, anchorSpawn, slack;
	Cond c[MAX_CONDS];
};

struct Hit {
	int64_t seed;
	int spawnX, spawnZ;
	int n;
	int x[MAX_HITS], z[MAX_HITS], what[MAX_HITS];
};

HD int64_t dist2(int ax, int az, int bx, int bz) {
	int64_t dx = ax - bx, dz = az - bz;
	return dx * dx + dz * dz;
}

// Stage A: placement-only, depends on the low 48 bits only. Counts candidate chunks of the condition's
// sets around the world origin, within radius+slack (the spawn is near but not at the origin).
HDN bool stageA(const StructTables& T, const Query& q, uint64_t seed) {
	for (int ci = 0; ci < q.ncond; ci++) {
		const Cond& c = q.c[ci];
		if (c.kind != COND_STRUCTURE) continue;
		int R = c.radius + (q.anchorSpawn ? q.slack : 0);
		int64_t R2 = (int64_t)R * R;
		int n = 0;
		for (int si = 0; si < T.nsets && n < c.count; si++) {
			if (!((c.setMask >> si) & 1) || T.sets[si].type != SET_SPREAD) continue;
			const SetDev& s = T.sets[si];
			int c0 = (-R - 8) >> 4, c1 = (R + 8) >> 4;
			int r0 = floorDiv(c0, s.spread.spacing), r1 = floorDiv(c1, s.spread.spacing);
			for (int rx = r0; rx <= r1 && n < c.count; rx++)
				for (int rz = r0; rz <= r1 && n < c.count; rz++) {
					int px, pz;
					spreadCandidate(s.spread, seed, rx, rz, px, pz);
					if (dist2(px * 16 + 8, pz * 16 + 8, 0, 0) > R2) continue;
					if (!frequencyOk(s.spread, seed, px, pz)) continue;
					if (s.exclusion >= 0 && !isStructureChunk(T, si, seed, px, pz)) continue;
					n++;
				}
		}
		if (n < c.count) return false;
	}
	return true;
}

// Stage B: full check for one 64-bit seed. w must be worldInit'ed for that seed.
HDN bool stageB(const World& w, const PackDev& pk, const StructTables& T, const Query& q, Hit& hit) {
	uint64_t seed = (uint64_t)w.seed;
	hit.seed = w.seed;
	hit.n = 0;
	int ax = 0, az = 0;
	// cheap necessary pass first: structure conditions around the origin with slack
	for (int ci = 0; ci < q.ncond; ci++) {
		const Cond& c = q.c[ci];
		if (c.kind != COND_STRUCTURE) continue;
		int R = c.radius + (q.anchorSpawn ? q.slack : 0);
		int64_t R2 = (int64_t)R * R;
		int n = 0;
		for (int si = 0; si < T.nsets && n < c.count; si++) {
			if (!((c.setMask >> si) & 1) || T.sets[si].type != SET_SPREAD) continue;
			const SetDev& s = T.sets[si];
			int c0 = (-R - 8) >> 4, c1 = (R + 8) >> 4;
			int r0 = floorDiv(c0, s.spread.spacing), r1 = floorDiv(c1, s.spread.spacing);
			for (int rx = r0; rx <= r1 && n < c.count; rx++)
				for (int rz = r0; rz <= r1 && n < c.count; rz++) {
					int px, pz;
					spreadCandidate(s.spread, seed, rx, rz, px, pz);
					if (dist2(px * 16 + 8, pz * 16 + 8, 0, 0) > R2 || !isStructureChunk(T, si, seed, px, pz)) continue;
					int st = structureAt(w, pk, T, si, px, pz);
					if (st >= 0 && ((c.structMask >> st) & 1)) n++;
				}
		}
		if (n < c.count) return false;
	}
	if (q.anchorSpawn) spawnOrigin(w, pk, ax, az);
	hit.spawnX = ax;
	hit.spawnZ = az;
	for (int ci = 0; ci < q.ncond; ci++) {
		const Cond& c = q.c[ci];
		int64_t R2 = (int64_t)c.radius * c.radius;
		if (c.kind == COND_SPAWN) {
			if (dist2(ax, az, 0, 0) > R2) return false;
		} else if (c.kind == COND_STRUCTURE) {
			int n = 0;
			for (int si = 0; si < T.nsets && n < c.count; si++) {
				if (!((c.setMask >> si) & 1) || T.sets[si].type != SET_SPREAD) continue;
				const SetDev& s = T.sets[si];
				int r0x = floorDiv((ax - c.radius - 8) >> 4, s.spread.spacing), r1x = floorDiv((ax + c.radius + 8) >> 4, s.spread.spacing);
				int r0z = floorDiv((az - c.radius - 8) >> 4, s.spread.spacing), r1z = floorDiv((az + c.radius + 8) >> 4, s.spread.spacing);
				for (int rx = r0x; rx <= r1x && n < c.count; rx++)
					for (int rz = r0z; rz <= r1z && n < c.count; rz++) {
						int px, pz;
						spreadCandidate(s.spread, seed, rx, rz, px, pz);
						if (dist2(px * 16 + 8, pz * 16 + 8, ax, az) > R2 || !isStructureChunk(T, si, seed, px, pz)) continue;
						int st = structureAt(w, pk, T, si, px, pz);
						if (st >= 0 && ((c.structMask >> st) & 1)) {
							n++;
							if (hit.n < MAX_HITS) {
								hit.x[hit.n] = px * 16 + 8;
								hit.z[hit.n] = pz * 16 + 8;
								hit.what[hit.n++] = st;
							}
						}
					}
			}
			if (n < c.count) return false;
		} else if (c.kind == COND_BIOME) {
			// surface-ish biome (y=64) on a 16-block grid, nearest rings first
			bool found = false;
			int steps = c.radius / 16;
			for (int r = 0; r <= steps && !found; r++)
				for (int dz = -r; dz <= r && !found; dz++)
					for (int dx = -r; dx <= r && !found; dx++) {
						if (dx != -r && dx != r && dz != -r && dz != r) continue;
						int bx = ax + dx * 16, bz = az + dz * 16;
						if (dist2(bx, bz, ax, az) > R2) continue;
						if (biomeAt(w, pk, bx >> 2, 16, bz >> 2, 0) == c.biome) {
							found = true;
							if (hit.n < MAX_HITS) {
								hit.x[hit.n] = bx;
								hit.z[hit.n] = bz;
								hit.what[hit.n++] = -1 - c.biome;
							}
						}
					}
			if (!found) return false;
		} else if (c.kind == COND_STRONGHOLD) {
			int sx[8], sz[8], n = 0;
			for (int si = 0; si < T.nsets; si++)
				if (T.sets[si].type == SET_RINGS) {
					int got = strongholds(w, pk, T.sets[si], 3, sx, sz);  // first ring
					for (int i = 0; i < got; i++)
						if (dist2(sx[i] * 16 + 4, sz[i] * 16 + 4, ax, az) <= R2) {
							n++;
							if (hit.n < MAX_HITS) {
								hit.x[hit.n] = sx[i] * 16 + 4;
								hit.z[hit.n] = sz[i] * 16 + 4;
								hit.what[hit.n++] = -1000;
							}
						}
				}
			if (n < c.count) return false;
		}
	}
	return true;
}

}  // namespace sw

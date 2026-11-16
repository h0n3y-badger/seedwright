// Structure placement + biome validation (device-capable tables built from the host Pack).
#pragma once
#include "sw_core.h"

namespace sw {

enum { MAX_SETS = 32, MAX_STRUCTS = 64, MAX_SET_STRUCTS = 32 };
enum SetType : uint8_t { SET_SPREAD, SET_RINGS, SET_OTHER };

struct SetDev {
	uint8_t type;
	Spread spread;
	int exclusion, exclusionChunks;
	int nstructs, structs[MAX_SET_STRUCTS], weights[MAX_SET_STRUCTS];
	// concentric rings
	int distance, ringSpread, count;
	uint64_t preferred[2];
};

enum { CHECK_CENTER, CHECK_FIXED_Y, CHECK_MINESHAFT };
struct StructDev {
	uint64_t biomes[2];
	int jigsaw, check, y;
};

struct StructTables {
	int nsets, nstructs;
	SetDev sets[MAX_SETS];
	StructDev st[MAX_STRUCTS];
};

HD bool maskHas(const uint64_t* m, int b) { return b >= 0 && ((m[b >> 6] >> (b & 63)) & 1); }

// StructurePlacement.isStructureChunk for random_spread sets (placement + frequency + exclusion zone)
HDN bool isStructureChunk(const StructTables& T, int set, uint64_t seed, int cx, int cz) {
	const SetDev& s = T.sets[set];
	if (s.type != SET_SPREAD) return false;
	if (!isSpreadChunk(s.spread, seed, cx, cz)) return false;
	if (s.exclusion >= 0) {
		const SetDev& o = T.sets[s.exclusion];
		int r = s.exclusionChunks;
		// only regions that can hold a candidate in range need checking
		int rx0 = floorDiv(cx - r, o.spread.spacing), rx1 = floorDiv(cx + r, o.spread.spacing);
		int rz0 = floorDiv(cz - r, o.spread.spacing), rz1 = floorDiv(cz + r, o.spread.spacing);
		for (int rx = rx0; rx <= rx1; rx++)
			for (int rz = rz0; rz <= rz1; rz++) {
				int px, pz;
				spreadCandidate(o.spread, seed, rx, rz, px, pz);
				if (px >= cx - r && px <= cx + r && pz >= cz - r && pz <= cz + r && frequencyOk(o.spread, seed, px, pz)) return false;
			}
	}
	return true;
}

// Column-volume climate for one quart column (couldStructureExistInColumn). y-independent ops are
// shared, so this evaluates the program once per y only for the ops downstream of the y gradient.
struct Column {
	int qminY, qmaxY;
	int biome[100];  // filled by columnBiomes; columnClimate leaves it -1 (lazy)
	float c6[100][6];
	HD float depth(int qy) const { return c6[qy - qminY][4]; }
};

HDN void columnClimate(const World& w, const PackDev& pk, int qx, int qz, Column& col) {
	col.qminY = (pk.minY - 1) >> 2;
	col.qmaxY = (pk.minY + pk.height - 1) >> 2;
	double pv[MAX_OPS], v[MAX_OPS];
	int upto = climateUpto(pk);
	for (int qy = col.qminY; qy <= col.qmaxY; qy++) {
		evalOpsState(w, pk, qx << 2, qy << 2, qz << 2, true, false, upto, pv, v, qy != col.qminY);
		for (int i = 0; i < 6; i++) col.c6[qy - col.qminY][i] = (float)v[pk.climate[i]];
		col.biome[qy - col.qminY] = -1;
	}
}

// bottom-up like vanilla's column loop, each search seeded with the previous leaf (RTree.lastResult)
HD int columnBiome(const PackDev& pk, Column& col, int qy, int& leaf) {
	int& b = col.biome[qy - col.qminY];
	if (b < 0) {
		leaf = leafFromClimate(pk, col.c6[qy - col.qminY], leaf);
		b = pk.rnodes[leaf].biome;
	}
	return b;
}

HDN void columnBiomes(const World& w, const PackDev& pk, int qx, int qz, Column& col) {
	int leaf = -1;
	col.qminY = (pk.minY - 1) >> 2;
	col.qmaxY = (pk.minY + pk.height - 1) >> 2;
	double pv[MAX_OPS], v[MAX_OPS];
	int upto = climateUpto(pk);
	for (int qy = col.qminY; qy <= col.qmaxY; qy++) {
		evalOpsState(w, pk, qx << 2, qy << 2, qz << 2, true, false, upto, pv, v, qy != col.qminY);
		float c6[6];
		for (int i = 0; i < 6; i++) c6[i] = (float)v[pk.climate[i]];
		leaf = leafFromClimate(pk, c6, leaf);
		col.biome[qy - col.qminY] = pk.rnodes[leaf].biome;
		for (int i = 0; i < 6; i++) col.c6[qy - col.qminY][i] = c6[i];
	}
}

// Biome validity of a structure start at chunk (cx,cz): column pre-check at the chunk middle (exact
// vanilla volume semantics), then the start-position check at an estimated surface height (highest
// quart with depth >= 0; vanilla uses the real terrain height there -> approximate until terrain lands).
HDN bool structureBiomeOk(const World& w, const PackDev& pk, const StructTables& T, int st, int cx, int cz) {
	const StructDev& S = T.st[st];
	if (S.check == CHECK_FIXED_Y)  // jigsaw with an absolute start height and no heightmap projection
		return maskHas(S.biomes, biomeAt(w, pk, ((cx << 4) + 8) >> 2, S.y >> 2, ((cz << 4) + 8) >> 2, 0));
	if (S.check == CHECK_MINESHAFT)  // (middleX, 50 + yOffset, minZ), no column pre-check
		return maskHas(S.biomes, biomeAt(w, pk, ((cx << 4) + 8) >> 2, S.y >> 2, (cz << 4) >> 2, 0));
	int qx = ((cx << 4) + 8) >> 2, qz = ((cz << 4) + 8) >> 2;
	Column col;
	columnClimate(w, pk, qx, qz, col);
	int surf = col.qminY;
	for (int qy = col.qmaxY; qy >= col.qminY; qy--)
		if (col.depth(qy) >= 0.0f) {
			surf = qy;
			break;
		}
	// both checks must pass: run the single-lookup start check first, then the column "any" scan
	// starting at the surface (any-of is order independent; RTree results don't depend on the seed leaf
	// except for exact ties, which vanilla itself resolves by thread history)
	if (!maskHas(S.biomes, biomeAt(w, pk, qx, surf, qz, 2))) return false;
	if (pk.legacy) return true;  // pre-26.3 has no column pre-check
	int leaf = -1;
	if (maskHas(S.biomes, columnBiome(pk, col, surf, leaf))) return true;
	for (int qy = col.qminY; qy <= col.qmaxY; qy++)
		if (qy != surf && maskHas(S.biomes, columnBiome(pk, col, qy, leaf))) return true;
	return false;
}

// Which structure of a set generates at a placement chunk (-1 = none): vanilla's weighted retry loop.
HDN int structureAt(const World& w, const PackDev& pk, const StructTables& T, int set, int cx, int cz) {
	const SetDev& s = T.sets[set];
	if (s.nstructs == 1) return structureBiomeOk(w, pk, T, s.structs[0], cx, cz) ? s.structs[0] : -1;
	JavaRandom r;
	largeFeatureRandom(r, (uint64_t)w.seed, cx, cz);
	int alive[MAX_SET_STRUCTS], n = s.nstructs, total = 0;
	for (int i = 0; i < n; i++) {
		alive[i] = i;
		total += s.weights[i];
	}
	while (n > 0) {
		int choice = r.nextInt(total), idx = 0;
		for (; idx < n; idx++) {
			choice -= s.weights[alive[idx]];
			if (choice < 0) break;
		}
		int pick = alive[idx];
		if (structureBiomeOk(w, pk, T, s.structs[pick], cx, cz)) return s.structs[pick];
		total -= s.weights[pick];
		for (int j = idx; j < n - 1; j++) alive[j] = alive[j + 1];
		n--;
	}
	return -1;
}

// ---- strongholds (ConcentricRingsStructurePlacement via ChunkGeneratorStructureState) ----
// The ring RNG never depends on biomes: each stronghold's raw ring chunk and its biome-search fork seed
// come from one cheap pass; the expensive biome search (moves it by <= 112 blocks) runs per stronghold.
HDN int strongholdRings(uint64_t seed, const SetDev& s, int want, int* rawX, int* rawZ, uint64_t* fork) {
	JavaRandom rnd;
	rnd.setSeed(seed);
	double angle = rnd.nextDouble() * 3.141592653589793 * 2.0;
	int pos = 0, circle = 0, spread = s.ringSpread, n = 0;
	for (int i = 0; i < s.count && n < want; i++) {
		double dist = (double)(4 * s.distance + s.distance * circle * 6) + (rnd.nextDouble() - 0.5) * (double)(s.distance * 2.5);
		rawX[n] = (int)(int64_t)floor(cos(angle) * dist + 0.5);  // Math.round
		rawZ[n] = (int)(int64_t)floor(sin(angle) * dist + 0.5);
		fork[n] = (uint64_t)rnd.nextLong();  // random.fork()
		n++;
		angle += 6.283185307179586 / spread;
		if (++pos == spread) {
			circle++;
			pos = 0;
			spread += 2 * spread / (circle + 1);
			spread = spread < s.count - i ? spread : s.count - i;
			angle += rnd.nextDouble() * 3.141592653589793 * 2.0;
		}
	}
	return n;
}

// findBiomeHorizontal(sectionToBlock(ix,8), 0, sectionToBlock(iz,8), 112, preferred, fork) -> final chunk
HDN void strongholdResolve(const World& w, const PackDev& pk, const SetDev& s, int ix, int iz, uint64_t forkSeed, int& outX, int& outZ) {
	JavaRandom search;
	search.setSeed(forkSeed);
	int cxq = ((ix << 4) + 8) >> 2, czq = ((iz << 4) + 8) >> 2, R = 112 >> 2;
	int found = 0, rx = 0, rz = 0, leaf = -1;
	float pv[MAX_OPS];
	int upto = climateUpto(pk);
	for (int dz = -R; dz <= R; dz++)
		for (int dx = -R; dx <= R; dx++) {
			// vanilla's RTree carries lastResult from query to query: chain it through the scan
			evalOps(w, pk, (cxq + dx) << 2, 0, (czq + dz) << 2, false, false, upto, pv);
			float c6[6];
			for (int k = 0; k < 6; k++) c6[k] = pv[pk.climate[k]];
			leaf = leafFromClimate(pk, c6, leaf);
			int b = pk.rnodes[leaf].biome;
			if (maskHas(s.preferred, b)) {
				if (found == 0 || search.nextInt(found + 1) == 0) {
					rx = (cxq + dx) << 2;
					rz = (czq + dz) << 2;
				}
				found++;
			}
		}
	outX = found ? rx >> 4 : ix;
	outZ = found ? rz >> 4 : iz;
}

// Computes the first `want` ring positions (each is independent of the later ones).
HDN int strongholds(const World& w, const PackDev& pk, const SetDev& s, int want, int* outX, int* outZ) {
	enum { CAP = 128 };
	int rx[CAP], rz[CAP];
	uint64_t fk[CAP];
	int n = strongholdRings((uint64_t)w.seed, s, want < CAP ? want : CAP, rx, rz, fk);
	for (int i = 0; i < n; i++) strongholdResolve(w, pk, s, rx[i], rz[i], fk[i], outX[i], outZ[i]);
	return n;
}

}  // namespace sw

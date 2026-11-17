#pragma once
// Warp-per-seed GPU stage B (kStageBWarp). The 32 lanes of a warp evaluate one seed together: every lane runs the
// query tree in lockstep with identical state (same seed, same decisions), and the hot loops below split their
// samples across lanes and combine the per-lane results in scan order, so every result (and every reported
// coordinate) is the one the sequential code gives:
//  - worldInit: noise layers initialised in parallel (independent RNG streams).
//  - spawn search: 32 spiral candidates per step, each bounded by the best fitness before the step; the step's
//    minimum (lowest index on ties) replaces the best iff it is strictly smaller (= the sequential update chain).
//  - biome conditions: 32 columns per step in ring order; biomeIs is a pure function of the climate (its witness
//    only prunes), so the first column (lowest lane) with a match is the sequential answer.
//  - stronghold biome search: per cell, the climate and the unchained RTree result (first leaf of globally minimal
//    distance) in parallel; vanilla's lastResult chain is then resolved in order: the chained search returns its
//    start leaf L iff L is at the global minimum distance, else the unchained result. Reservoir pick and the
//    in/out cut-offs run in order on the chained leaves, as in strongholdResolve.
#include "query.h"

#ifndef SW_CPU_ONLY
namespace sw {

__device__ __forceinline__ int laneId() { return threadIdx.x & 31; }
#define WFULL 0xffffffffu

__device__ void worldInitWarp(World& w, const PackDev& pk, int64_t seed) {
	Xoro r;
	r.seed64((uint64_t)seed);
	uint64_t posLo = r.nextLong(), posHi = r.nextLong();
	for (int li = laneId(); li < pk.nlayers; li += 32) {
		int n = 0;
		while (n + 1 < pk.nnoises && pk.noises[n + 1].layer0 <= li) n++;
		const NoiseDesc& nd = pk.noises[n];
		if (li < nd.layer0 || li >= nd.layer0 + nd.nlayers) continue;  // layer of no noise
		Xoro nr;
		nr.init(nd.hlo ^ posLo, nd.hhi ^ posHi);
		uint64_t f[4];
		for (int k = 0; k < 4; k++) f[k] = nr.nextLong();
		const LayerDesc& L = pk.layers[li];
		Xoro lr;
		lr.init(L.hlo ^ f[L.second * 2], L.hhi ^ f[L.second * 2 + 1]);
		perlinInit(w.perlin[li], lr);
	}
	if (laneId() == 0) w.seed = seed;
	__syncwarp();
}

// min of (f, idx) over the warp, lowest idx on ties; result in every lane
__device__ __forceinline__ void warpArgMin(int64_t& f, int& idx) {
	for (int o = 16; o > 0; o >>= 1) {
		int64_t of = __shfl_xor_sync(WFULL, f, o);
		int oi = __shfl_xor_sync(WFULL, idx, o);
		if (of < f || (of == f && oi < idx)) f = of, idx = oi;
	}
}

__device__ void spawnOriginWarp(const World& w, const PackDev& pk, int& outX, int& outZ) {
	int lane = laneId();
	int bx = 0, bz = 0;
	int64_t best = spawnFitnessBounded(w, pk, pk.spawnPlan, 0, 0, INT64_MAX);
	for (int s = 0; s < 2; s++) {
		int ox = bx, oz = bz, n = pk.nspiral[s];
		for (int b = 0; b < n; b += 32) {
			int i = b + lane;
			int64_t f = INT64_MAX;
			if (i < n) {
				int x = ox + pk.spiral[s][i][0], z = oz + pk.spiral[s][i][1];
				f = spawnDistancePenalty(pk, x, z) >= best ? INT64_MAX : spawnFitnessBounded(w, pk, pk.spawnPlan, x, z, best);
			}
			int idx = i < n ? i : INT32_MAX;
			warpArgMin(f, idx);
			if (f < best) {
				best = f;
				bx = ox + pk.spiral[s][idx][0];
				bz = oz + pk.spiral[s][idx][1];
			}
		}
	}
	outX = bx;
	outZ = bz;
}

__device__ bool evalBiomeWarp(EvalCtx& c, const Node& nd, int ax, int az, int depth) {
	int64_t R2 = (int64_t)nd.radius * nd.radius;
	int steps = nd.radius / 16;
	int y0 = nd.y == Y_ANY ? -48 : nd.y, y1 = nd.y == Y_ANY ? 48 : nd.y;
	const PackDev& pk = *c.pk;
	int upto = climateUpto(pk), wit = -1;
	bool early = nd.nkc[0] > 0 || nd.nkc[1] > 0;
	int lane = laneId(), k = 0, myX = 0, myZ = 0;
	// lanes < cnt test their column (all y levels, in order); true + hit if any matched
	auto step = [&](int cnt) -> bool {
		bool hit = false;
		if (lane < cnt) {
			double pv[MAX_OPS], vv[MAX_OPS];
			for (int y = y0; y <= y1 && !hit; y += 16) {
				int qx = (myX >> 2) << 2, qy = (y >> 2) << 2, qz = (myZ >> 2) << 2;
				if (y == y0 && early) {
					uint64_t done = 0;
					bool rej = false;
					for (int s = 0; s < 2 && !rej; s++) {
						if (!nd.nkc[s]) continue;
						evalOpsState(*c.w, pk, qx, qy, qz, false, false, upto, pv, vv, false, c.q->opsK[s] & ~done);
						done |= c.q->opsK[s];
						int64_t tk[3];
						for (int i = 0; i <= s + 1; i++) tk[i] = quantize((float)pv[pk.climate[i]]);
						rej = kReject(*c.q, nd, s, tk);
					}
					if (rej) break;
					evalOpsState(*c.w, pk, qx, qy, qz, false, false, upto, pv, vv, false, ~done);
				} else
					evalOpsState(*c.w, pk, qx, qy, qz, false, false, upto, pv, vv, y != y0);
				int64_t t[7];
				for (int i = 0; i < 6; i++) t[i] = quantize((float)pv[pk.climate[i]]);
				t[6] = 0;
				hit = biomeIs(pk, *c.q, nd, t, wit);
			}
		}
		unsigned b = __ballot_sync(WFULL, hit);
		if (!b) return false;
		int src = __ffs(b) - 1;
		int hx = __shfl_sync(WFULL, myX, src), hz = __shfl_sync(WFULL, myZ, src);
		addHit(c, -1 - nd.biome, hx, hz, depth);
		return true;
	};
	for (int r = 0; r <= steps; r++)
		for (int dz = -r; dz <= r; dz++)
			for (int dx = -r; dx <= r; dx += (dz == -r || dz == r) ? 1 : 2 * r) {
				int bx = ax + dx * 16, bz = az + dz * 16;
				if (dist2(bx, bz, ax, az) > R2) continue;
				if (k == lane) myX = bx, myZ = bz;
				if (++k == 32) {
					if (step(32)) return true;
					k = 0;
				}
			}
	return k > 0 && step(k);
}

__device__ int strongholdResolveWarp(const World& w, const PackDev& pk, const SetDev& s, int ix, int iz, uint64_t forkSeed, int& outX,
                                     int& outZ, int ax, int az, int64_t r2) {
	JavaRandom search;
	search.setSeed(forkSeed);
	int cxq = ((ix << 4) + 8) >> 2, czq = ((iz << 4) + 8) >> 2, R = 112 >> 2, W = 2 * R + 1;
	const int N = W * W;
	int found = 0, rx = 0, rz = 0, leaf = -1, done = 0, nextWin = 0, lastRow = -1;
	bool pickMoved = true;
	int upto = climateUpto(pk), lane = laneId();
	const int bx0 = (((cxq - R) << 2) >> 4) * 16 + 4, bx1 = (((cxq + R) << 2) >> 4) * 16 + 4;
	for (int b = 0; b < N; b += 32) {
		// this lane's cell: climate + unchained search (first leaf at the global minimum distance)
		int idx = b + lane;
		int64_t t[7] = {0, 0, 0, 0, 0, 0, 0}, minD = 0;
		int first = 0;
		if (idx < N) {
			float pv[MAX_OPS];
			evalOps(w, pk, (cxq - R + idx % W) << 2, 0, (czq - R + idx / W) << 2, false, false, upto, pv);
			for (int k = 0; k < 6; k++) t[k] = quantize(pv[pk.climate[k]]);
			first = rtreeSearch(pk, 0, t, -1);
			minD = nodeDist(pk.rnodes[first], t);
		}
		int cnt = N - b < 32 ? N - b : 32;
		for (int k = 0; k < cnt; k++) {
			// chained search from `leaf`: stays on it iff it is at the global minimum, else the unchained result
			int64_t dl = leaf >= 0 ? nodeDist(pk.rnodes[leaf], t) : -1;
			int64_t dk = __shfl_sync(WFULL, dl, k), mk = __shfl_sync(WFULL, minD, k);
			int fk = __shfl_sync(WFULL, first, k);
			leaf = (leaf >= 0 && dk == mk) ? leaf : fk;
			done++;
			int dx = (b + k) % W - R, dz = (b + k) / W - R;
			if (maskHas(s.preferred, pk.rnodes[leaf].biome)) {
				if (found == nextWin) {
					rx = (cxq + dx) << 2;
					rz = (czq + dz) << 2;
					pickMoved = true;
					int j = found + 1, lim = found + 1 + (N - done);
					while (j < lim && search.nextInt(j + 1) != 0) j++;
					nextWin = j;
				}
				found++;
			}
			if (nextWin >= found + (N - done)) goto final;
			if (r2 >= 0) {
				int row = (done + nextWin - found) / W;
				if (row != lastRow || pickMoved) {
					lastRow = row;
					pickMoved = false;
					int64_t cx = (found ? rx >> 4 : ix) * 16 + 4 - ax, cz = (found ? rz >> 4 : iz) * 16 + 4 - az;
					bool in = cx * cx + cz * cz <= r2;
					int64_t dmin2, dmax2;
					boxDist2(bx0, bx1, (((czq - R + row) << 2) >> 4) * 16 + 4, (((czq + R) << 2) >> 4) * 16 + 4, ax, az, dmin2, dmax2);
					if (in && dmax2 <= r2) return SH_IN;
					if (!in && dmin2 > r2) return SH_OUT;
				}
			}
		}
	}
final:
	outX = found ? rx >> 4 : ix;
	outZ = found ? rz >> 4 : iz;
	return SH_EXACT;
}

// One warp per seed (see the top of this file). worlds: one World per warp.
__global__ void __launch_bounds__(128) kStageBWarp(const PackDev* pk, const StructTables* T, const Query* q, const int64_t* seeds, unsigned n,
                                                  World* worlds, Hit* hits, unsigned* nhits, unsigned cap, const JigsawTables* J, int prefilter,
                                                  const LootTables* L, bool skipPre) {
	unsigned wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, nw = (gridDim.x * blockDim.x) >> 5;
	// World in global memory: a shared-memory World (13 KB per warp) was 1.3-2.5x slower (fewer resident warps)
	World& w = worlds[wid];
	for (unsigned i = wid; i < n; i += nw) {
		worldInitWarp(w, *pk, seeds[i]);
		Hit h;
		bool ok = stageB(w, *pk, *T, *q, h, J, nullptr, nullptr, prefilter != 0, MAX_PIECES, L, true, skipPre ? 2 : 0);  // skipPre: kStagePre ran
		if (ok && laneId() == 0) {
			unsigned k = atomicAdd(nhits, 1u);
			if (k < cap) hits[k] = h;
		}
		__syncwarp();  // the next seed's init overwrites w
	}
}

// Thread per seed: the cheap PRE pass (structure placement + biome validity at 0,0 with slack, mostly single-lane
// work) for kStageBWarp; survivors are compacted into out.
__global__ void __launch_bounds__(128, 4) kStagePre(const PackDev* pk, const StructTables* T, const Query* q, const int64_t* seeds, unsigned n,
                                                   World* worlds, const JigsawTables* J, const LootTables* L, int64_t* out, unsigned* nout) {
	World& w = worlds[blockIdx.x * blockDim.x + threadIdx.x];
	for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
		worldInit(w, *pk, seeds[i]);
		Hit h;
		if (stageB(w, *pk, *T, *q, h, J, nullptr, nullptr, false, MAX_PIECES, L, false, 1)) out[atomicAdd(nout, 1u)] = seeds[i];
	}
}

}  // namespace sw
#endif

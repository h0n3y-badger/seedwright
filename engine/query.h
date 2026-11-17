// Seed filters as a small condition tree ("2 plains villages within 500 { 1 weaponsmith }, biome
// cherry_grove within 1500, not monument within 800"). Device-capable; nodes live in a flat array.
#pragma once
#include "structures.h"
#include "jigsaw.h"
#include "loot.h"
#include <vector>

namespace sw {

enum NodeKind {
	N_ALL,         // every child holds
	N_ANY,         // at least one child holds
	N_NOT,         // no child holds (children are an implicit ALL)
	N_STRUCTURE,   // >= count structures within radius of the anchor, each satisfying the children
	N_BIOME,       // biome within radius of the anchor at height y (or any of the cave levels)
	N_STRONGHOLD,  // >= count strongholds within radius, each satisfying the children
	N_SPAWN,       // world spawn within radius of 0,0 (independent of the anchor)
	N_BUILDING,    // >= count buildings of a category in the enclosing village (exact with terrain.h heights, see stageB)
	N_LOOT,        // >= count of an item in the enclosing structure's loot containers (loot.h; villages: like N_BUILDING)
};

enum { MAX_NODES = 48, MAX_HITS = 32, MAX_DEPTH = 6, Y_ANY = -100000, MAX_STRONGHOLDS = 32, MAX_PIECES = 320, MAX_BIOME_LEAVES = 4096, MAX_KCOND = 8, MAX_KB = 512 };

struct Node {
	int kind, count, radius, y;
	int first, next;      // first child, next sibling (-1 = none)
	uint64_t structMask;  // N_STRUCTURE: structure indices (bit i = structure i)
	uint32_t setMask;     // sets that can produce one of those structures
	int biome;            // N_BIOME
	int leaf0, nleaf;     // N_BIOME: RTree leaves of that biome in Query::leaves (nleaf 0 = no list, always full search)
	int64_t bbLo[7], bbHi[7];  // N_BIOME: union box of those leaves
	// N_BIOME early reject from the cheap climate dims (see prepareKReject):
	// "no" when for every c < nkc: min over K boxes in kcO[c] of the K distance < min over K boxes in kcW[c]
	// stage s: K boxes Query::kb[kb0[s], +nkb[s]) (masks index them)
	int nkc[2], kb0[2], nkb[2];
	uint64_t kcW[2][MAX_KCOND], kcO[2][MAX_KCOND];
	int item, lootJig;    // N_LOOT: item (LootHost::items), 1 = needs a jigsaw layout (villages), 0 = buried treasure
	uint64_t cat[3];      // N_BUILDING: jigsaw building categories that count (aliases: weaponsmith|weapon_smith)
};

struct Query {
	int nnodes, root, anchorSpawn, slack, needSpawn;
	int hasBuildings;  // needs village layouts (jigsaw tables + a piece buffer)
	int hasVillageLoot;  // village loot conditions want the whole layout: building checks then don't stop it early
	int exact;  // skip the slack-based prefilter (checking given seeds: the prefilter assumes spawn within slack of 0,0)
	// building prefilter (stageB with prefilter=true, estimated terrain; survivors are re-checked with exact heights):
	int buildingSlack;  // estimated layout needs count - buildingSlack buildings (estimates miss real ones)
	int buildingPass;   // 1: building conditions don't filter at all there (exhaustive: every candidate is laid out exactly)
	Node n[MAX_NODES];
	int nleaves;
	uint16_t leaves[MAX_BIOME_LEAVES];
	// early-reject K boxes over climate dims {0, 1, 2, 6} (see prepareKReject); opsK[s]: ops computing stage s's dims
	int nkb;
	int64_t kbLo[MAX_KB][4], kbHi[MAX_KB][4];
	uint64_t opsK[2];
};

// Host: the early reject from cheap climate dims. Stage 0 knows K = {temperature, humidity} (4+4 noise octaves
// after the shifts), stage 1 adds continentalness (18 more); erosion and weirdness (14) and depth come after.
// For a wanted leaf w and a non-wanted leaf o whose box contains w's box in every dim outside K, o is at least as
// close as w in those dims for any climate, so dK(o) < dK(w) proves d(o) < d(w) whatever the remaining noises
// give. When every wanted leaf is beaten that way, the winner (a leaf of minimal distance) can't be the wanted
// biome. Per wanted leaf: (its K box, the K boxes of such o); equal o sets merge, implied conditions drop.
// K boxes live in Query::kbLo/kbHi over dims KDIMS (unknown dims of a stage get an unbounded range).
inline void prepareKReject(Query& q, const PackDev& pk) {
	const int KDIMS[4] = {0, 1, 2, 6};
	const int64_t INF = 1LL << 60;
	q.nkb = 0;
	for (int ni = 0; ni < q.nnodes; ni++) q.n[ni].nkc[0] = q.n[ni].nkc[1] = 0;
	int upto = climateUpto(pk);
	uint64_t dep[MAX_OPS];  // ops each op reads, transitively (incl. itself)
	for (int i = 0; i <= upto; i++) {
		const Op& o = pk.ops[i];
		uint64_t m = 1ULL << i;
		if (o.a >= 0) m |= dep[o.a];
		if (o.b >= 0) m |= dep[o.b];
		if (o.c >= 0) m |= dep[o.c];
		if (o.code == OP_SPLINE)
			for (int sp = 0; sp < pk.nsplines; sp++)  // any spline coordinate (over-approximation: still closed)
				if (pk.splines[sp].coord >= 0 && pk.splines[sp].coord < i) m |= dep[pk.splines[sp].coord];
		dep[i] = m;
	}
	q.opsK[0] = dep[pk.climate[0]] | dep[pk.climate[1]];
	q.opsK[1] = q.opsK[0] | dep[pk.climate[2]];
	for (int s = 0; s < 2; s++) {
		bool yind = true;  // a stage's dims must be y-independent (a column is rejected at once)
		for (int i = 0; i <= upto; i++) yind = yind && !(((q.opsK[s] >> i) & 1) && pk.ops[i].ydep);
		for (int ni = 0; ni < q.nnodes && yind; ni++) {
			Node& nd = q.n[ni];
			if (nd.kind != N_BIOME) continue;
			bool known[7] = {s >= 0, s >= 0, s >= 1, false, false, false, true};
			int kb0 = q.nkb, nkb = 0;
			auto boxOf = [&](const RNode& r) {  // local K-box index (dedup), -1 = pool full
				for (int k = 0; k < nkb; k++) {
					bool same = true;
					for (int j = 0; j < 4; j++)
						same = same && (!known[KDIMS[j]] || (q.kbLo[kb0 + k][j] == r.lo[KDIMS[j]] && q.kbHi[kb0 + k][j] == r.hi[KDIMS[j]]));
					if (same) return k;
				}
				if (nkb == 64 || kb0 + nkb == MAX_KB) return -1;
				for (int j = 0; j < 4; j++) {
					bool kn = known[KDIMS[j]];
					q.kbLo[kb0 + nkb][j] = kn ? r.lo[KDIMS[j]] : -INF;
					q.kbHi[kb0 + nkb][j] = kn ? r.hi[KDIMS[j]] : INF;
				}
				return nkb++;
			};
			std::vector<std::pair<uint64_t, uint64_t>> c;  // (W, O) local K-box masks
			bool fail = false;
			for (int w = 0; w < pk.nrnodes && !fail; w++) {
				const RNode& W = pk.rnodes[w];
				if (W.nchild != 0 || W.biome != nd.biome) continue;
				uint64_t om = 0;
				for (int o = 0; o < pk.nrnodes && !fail; o++) {
					const RNode& O = pk.rnodes[o];
					if (O.nchild != 0 || O.biome == nd.biome) continue;
					bool in = true;
					for (int d = 0; d < 7 && in; d++) in = known[d] || (O.lo[d] <= W.lo[d] && O.hi[d] >= W.hi[d]);
					if (!in) continue;
					int k = boxOf(O);
					if (k < 0) fail = true;
					else om |= 1ULL << k;
				}
				int kw = fail ? -1 : boxOf(W);
				if (om == 0 || kw < 0) fail = true;  // never provably beaten (or too many boxes)
				if (fail) break;
				bool merged = false;
				for (auto& e : c)
					if (e.second == om) e.first |= 1ULL << kw, merged = true;
				if (!merged) c.push_back({1ULL << kw, om});
			}
			if (fail || c.empty()) continue;
			std::vector<std::pair<uint64_t, uint64_t>> keep;  // drop (W, O) implied by some (W' >= W, O' <= O)
			for (size_t i = 0; i < c.size(); i++) {
				bool red = false;
				for (size_t j = 0; j < c.size() && !red; j++)
					red = c[j] != c[i] && (c[i].first & ~c[j].first) == 0 && (c[j].second & ~c[i].second) == 0;
				bool dup = false;
				for (auto& e : keep) dup = dup || e == c[i];
				if (!red && !dup) keep.push_back(c[i]);
			}
			if (keep.size() > MAX_KCOND) continue;
			q.nkb += nkb;
			nd.kb0[s] = kb0;
			nd.nkb[s] = nkb;
			nd.nkc[s] = (int)keep.size();
			for (int k = 0; k < nd.nkc[s]; k++) nd.kcW[s][k] = keep[k].first, nd.kcO[s][k] = keep[k].second;
		}
	}
}

// Host: per N_BIOME node, the RTree leaves that carry its biome (for the exact reject in evalBiome).
inline void prepareBiomeLeaves(Query& q, const PackDev& pk) {
	q.nleaves = 0;
	for (int ni = 0; ni < q.nnodes; ni++) {
		Node& nd = q.n[ni];
		nd.leaf0 = q.nleaves;
		nd.nleaf = 0;
		if (nd.kind != N_BIOME) continue;
		int n = 0;
		for (int i = 0; i < pk.nrnodes; i++)
			if (pk.rnodes[i].nchild == 0 && pk.rnodes[i].biome == nd.biome) n++;
		if (n == 0 || q.nleaves + n > MAX_BIOME_LEAVES || pk.nrnodes > 65536) continue;  // no list: full search
		for (int k = 0; k < 7; k++) nd.bbLo[k] = INT64_MAX, nd.bbHi[k] = INT64_MIN;
		for (int i = 0; i < pk.nrnodes; i++) {
			const RNode& r = pk.rnodes[i];
			if (r.nchild != 0 || r.biome != nd.biome) continue;
			q.leaves[q.nleaves++] = (uint16_t)i;
			for (int k = 0; k < 7; k++) {
				nd.bbLo[k] = r.lo[k] < nd.bbLo[k] ? r.lo[k] : nd.bbLo[k];
				nd.bbHi[k] = r.hi[k] > nd.bbHi[k] ? r.hi[k] : nd.bbHi[k];
			}
		}
		nd.nleaf = n;
	}
}

struct Hit {
	int64_t seed;
	int spawnX, spawnZ;
	int n;
	int x[MAX_HITS], z[MAX_HITS], what[MAX_HITS], depth[MAX_HITS];
};
enum { WHAT_STRONGHOLD = -1000, WHAT_SPAWN = -1001, WHAT_BUILDING0 = -2000, WHAT_LOOT0 = -100000 };  // biomes: -1 - biome; loot: WHAT_LOOT0 - (item * 64 + min(count, 63))

HD int64_t dist2(int ax, int az, int bx, int bz) {
	int64_t dx = ax - bx, dz = az - bz;
	return dx * dx + dz * dz;
}

// Evaluation modes. All three agree on the tree; A and PRE are sound over-approximations of B.
//  EVAL_A   : low 48 bits only, structure placement candidates (no biomes); non-structure leaves pass.
//  EVAL_PRE : full seed, structure placement + biome validity, top level only, before the spawn is known.
//  EVAL_B   : exact.
enum { EVAL_A, EVAL_PRE, EVAL_B };

struct EvalCtx {
	bool warp;  // GPU kStageBWarp: the 32 lanes of a warp evaluate one seed in lockstep (warp.h hot loops)
	const World* w;
	const PackDev* pk;
	const StructTables* T;
	const Query* q;
	uint64_t seed;
	int mode;
	Hit* hit;
	// strongholds: raw ring positions (cheap, all at once) and biome-resolved positions (lazy, per index)
	int shRaw, shRawX[MAX_STRONGHOLDS], shRawZ[MAX_STRONGHOLDS];
	uint64_t shFork[MAX_STRONGHOLDS];
	uint32_t shDone;
	int shX[MAX_STRONGHOLDS], shZ[MAX_STRONGHOLDS];
	int spawnX, spawnZ;
	int curStructure, curCx, curCz;  // innermost enclosing structure (for buildings)
	const JigsawTables* J;           // null: old pack, plain structureAt
	Piece* pcs;                      // MAX_PIECES layout buffer (global memory on the GPU)
	int npcs, layoutSt, layoutCx, layoutCz;
	bool layoutPartial;  // c.pcs is a prefix of the layout (stopped once a building condition was met)
	const HeightGrid* hg;  // exact terrain heights (host, terrain.h); null: estimateSurfaceY
	int prefilter;         // building conditions are a necessary-condition prefilter (see Query.buildingSlack)
	int maxPcs;            // size of pcs
	const LootTables* L;   // chest loot tables (null: no loot section)
};

HDN bool hasBuildingBelow(const Query* q, int ni) {
	for (int k = q->n[ni].first; k >= 0; k = q->n[k].next)
		if (q->n[k].kind == N_BUILDING || (q->n[k].kind == N_LOOT && q->n[k].lootJig) || hasBuildingBelow(q, k)) return true;
	return false;
}

HDN bool evalNode(EvalCtx& c, int ni, int ax, int az, int slack, int depth);

HDN bool evalChildren(EvalCtx& c, int ni, int ax, int az, int slack, int depth) {
	for (int k = c.q->n[ni].first; k >= 0; k = c.q->n[k].next)
		if (!evalNode(c, k, ax, az, slack, depth)) return false;
	return true;
}

HD void addHit(EvalCtx& c, int what, int x, int z, int depth) {
	Hit* h = c.hit;
	if (!h || h->n >= MAX_HITS) return;
	h->x[h->n] = x;
	h->z[h->n] = z;
	h->what[h->n] = what;
	h->depth[h->n++] = depth;
}

HDN bool evalStructure(EvalCtx& c, const Node& nd, int ni, int ax, int az, int slack, int depth) {
	int R = nd.radius + slack;
	int64_t R2 = (int64_t)R * R;
	int n = 0;
	for (int si = 0; si < c.T->nsets && n < nd.count; si++) {
		if (!((nd.setMask >> si) & 1) || c.T->sets[si].type != SET_SPREAD) continue;
		const SetDev& s = c.T->sets[si];
		int r0x = floorDiv((ax - R - 8) >> 4, s.spread.spacing), r1x = floorDiv((ax + R + 8) >> 4, s.spread.spacing);
		int r0z = floorDiv((az - R - 8) >> 4, s.spread.spacing), r1z = floorDiv((az + R + 8) >> 4, s.spread.spacing);
		for (int rx = r0x; rx <= r1x && n < nd.count; rx++)
			for (int rz = r0z; rz <= r1z && n < nd.count; rz++) {
				int px, pz;
				spreadCandidate(s.spread, c.seed, rx, rz, px, pz);
				int bx = px * 16 + 8, bz = pz * 16 + 8;
				if (dist2(bx, bz, ax, az) > R2) continue;
				if (c.mode == EVAL_A) {
					if (!frequencyOk(s.spread, c.seed, px, pz)) continue;
					if (s.exclusion >= 0 && !isStructureChunk(*c.T, si, c.seed, px, pz)) continue;
					if (evalChildren(c, ni, bx, bz, 0, depth + 1)) n++;
					continue;
				}
				if (!isStructureChunk(*c.T, si, c.seed, px, pz)) continue;
				int st = c.J ? jigsawStructureAt(*c.w, *c.pk, *c.T, *c.J, si, px, pz, c.hg) : structureAt(*c.w, *c.pk, *c.T, si, px, pz);
				if (st < 0 || !((nd.structMask >> st) & 1)) continue;
				if (c.mode == EVAL_PRE) {
					n++;
					continue;
				}
				int mark = c.hit ? c.hit->n : 0;
				addHit(c, st, bx, bz, depth);
				// children see the structure as their anchor (and the village id for buildings)
				int saveSt = c.curStructure, saveCx = c.curCx, saveCz = c.curCz;
				c.curStructure = st, c.curCx = px, c.curCz = pz;
				bool ok = evalChildren(c, ni, bx, bz, 0, depth + 1);
				c.curStructure = saveSt, c.curCx = saveCx, c.curCz = saveCz;
				if (ok) n++;
				else if (c.hit) c.hit->n = mark;
			}
	}
	return n >= nd.count;
}

// Is the biome at climate t the node's biome? Exact: the RTree search returns a leaf of globally minimal
// distance, so when some leaf W (the previous sample's result) is strictly closer than every leaf of the
// wanted biome, the result can't be of that biome, whatever the search's tie-breaking. Otherwise (or
// without a leaf list) the full vanilla search decides. wit: in/out previous full-search result.
HDN bool biomeIs(const PackDev& pk, const Query& q, const Node& nd, const int64_t* t, int& wit) {
	if (wit < 0 || pk.rnodes[0].nchild == 0) {
		wit = rtreeSearch(pk, 0, t, -1);
		return pk.rnodes[wit].biome == nd.biome;
	}
	// any leaf's distance bounds the winner's: dW (previous result W) seeds the search below
	int64_t dW = nodeDist(pk.rnodes[wit], t), stop = 0;
	if (nd.nleaf > 0) {
		int64_t lb = 0;  // lower bound of every wanted leaf's distance
		for (int k = 0; k < 7; k++) {
			int64_t d = pdist(nd.bbLo[k], nd.bbHi[k], t[k]);
			lb += d * d;
		}
		if (lb > dW) return false;  // W (not wanted, as lb > dW) is strictly closer than every wanted leaf
		if (nd.nleaf <= 128) {  // exact nearest wanted leaf; for big biomes (plains, desert) the search is cheaper
			int64_t mW = INT64_MAX;
			for (int i = 0; i < nd.nleaf; i++) {
				int64_t d = nodeDist(pk.rnodes[q.leaves[nd.leaf0 + i]], t);
				mW = d < mW ? d : mW;
			}
			if (mW > dW) return false;
			lb = mW;
			dW = mW;  // the winner's distance is <= mW too
		}
		stop = lb;  // a leaf closer than every wanted leaf decides "no" (wit then is that leaf, still a valid bound)
	}
	int best = -1;
	int64_t minD = dW + 1;
	rtreeSearchB(pk, 0, t, best, minD, stop);
	wit = best;
	return minD >= stop && pk.rnodes[best].biome == nd.biome;
}

// early reject (prepareKReject), stage s: true = the biome at a climate with these dims 0..2 (t[6] = 0) isn't nd's
HDN bool kReject(const Query& q, const Node& nd, int s, const int64_t* t) {
	int64_t dk[64];
	for (int k = 0; k < nd.nkb[s]; k++) {
		const int64_t *lo = q.kbLo[nd.kb0[s] + k], *hi = q.kbHi[nd.kb0[s] + k];
		int64_t d = 0;
		for (int j = 0; j < 4; j++) {
			int64_t e = pdist(lo[j], hi[j], j < 3 ? t[j] : 0);
			d += e * e;
		}
		dk[k] = d;
	}
	for (int c = 0; c < nd.nkc[s]; c++) {
		int64_t mW = INT64_MAX, mO = INT64_MAX;
		for (int k = 0; k < nd.nkb[s]; k++) {
			if ((nd.kcW[s][c] >> k) & 1) mW = dk[k] < mW ? dk[k] : mW;
			if ((nd.kcO[s][c] >> k) & 1) mO = dk[k] < mO ? dk[k] : mO;
		}
		if (!(mO < mW)) return false;
	}
	return true;
}

#ifdef __CUDA_ARCH__
__device__ bool evalBiomeWarp(EvalCtx& c, const Node& nd, int ax, int az, int depth);
__device__ int strongholdResolveWarp(const World& w, const PackDev& pk, const SetDev& s, int ix, int iz, uint64_t forkSeed, int& outX,
                                     int& outZ, int ax, int az, int64_t r2);
__device__ void spawnOriginWarp(const World& w, const PackDev& pk, int& outX, int& outZ);
#endif
HD int shResolveAny(const EvalCtx& c, const SetDev& s, int ix, int iz, uint64_t forkSeed, int& outX, int& outZ, int ax = 0, int az = 0,
                    int64_t r2 = -1) {
#ifdef __CUDA_ARCH__
	if (c.warp) return strongholdResolveWarp(*c.w, *c.pk, s, ix, iz, forkSeed, outX, outZ, ax, az, r2);
#endif
	return strongholdResolve(*c.w, *c.pk, s, ix, iz, forkSeed, outX, outZ, ax, az, r2);
}

HDN bool evalBiome(EvalCtx& c, const Node& nd, int ax, int az, int depth) {
#ifdef __CUDA_ARCH__
	if (c.warp) return evalBiomeWarp(c, nd, ax, az, depth);
#endif
	// on a 16-block grid, nearest rings first; y=any tries the cave levels (y -48..48, every 16).
	// Same samples and order as biomeAt(..., 0); y-independent climate ops are evaluated once per column.
	int64_t R2 = (int64_t)nd.radius * nd.radius;
	int steps = nd.radius / 16;
	int y0 = nd.y == Y_ANY ? -48 : nd.y, y1 = nd.y == Y_ANY ? 48 : nd.y;
	const PackDev& pk = *c.pk;
	int upto = climateUpto(pk), wit = -1;
	bool early = nd.nkc[0] > 0 || nd.nkc[1] > 0;
	double pv[MAX_OPS], vv[MAX_OPS];
	for (int r = 0; r <= steps; r++)
		for (int dz = -r; dz <= r; dz++)
			for (int dx = -r; dx <= r; dx += (dz == -r || dz == r) ? 1 : 2 * r) {  // ring cells only
				int bx = ax + dx * 16, bz = az + dz * 16;
				if (dist2(bx, bz, ax, az) > R2) continue;
				for (int y = y0; y <= y1; y += 16) {
					int qx = (bx >> 2) << 2, qy = (y >> 2) << 2, qz = (bz >> 2) << 2;
					if (y == y0 && early) {  // cheap dims first: an early reject holds for the whole column
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
					if (biomeIs(pk, *c.q, nd, t, wit)) {
						addHit(c, -1 - nd.biome, bx, bz, depth);
						return true;
					}
				}
			}
	return false;
}

HD const SetDev* ringSet(const StructTables& T) {
	for (int si = 0; si < T.nsets; si++)
		if (T.sets[si].type == SET_RINGS) return &T.sets[si];
	return nullptr;
}

// Strongholds within radius (+slack) of the anchor. A stronghold ends up inside a fixed box around its raw ring
// chunk (strongholdBox), so only those whose box straddles the radius get the expensive biome search, and that
// search stops as soon as the side is settled; PRE mode decides on the boxes alone.
HDN void shResolve(EvalCtx& c, const SetDev& ring, int i) {
	if ((c.shDone >> i) & 1) return;
	shResolveAny(c, ring, c.shRawX[i], c.shRawZ[i], c.shFork[i], c.shX[i], c.shZ[i]);
	c.shDone |= 1u << i;
}

HDN bool evalStronghold(EvalCtx& c, const Node& nd, int ni, int ax, int az, int slack, int depth) {
	const SetDev* ring = ringSet(*c.T);
	if (!ring) return false;
	if (c.shRaw < 0) c.shRaw = strongholdRings(c.seed, *ring, MAX_STRONGHOLDS, c.shRawX, c.shRawZ, c.shFork);
	int64_t R = nd.radius + slack, R2 = R * R;
	bool children = nd.first >= 0;
	// decide without biome searches where the box settles it: certainly in, certainly out; only borderline
	// ones are searched (and only until the side is known). Children need exact positions.
	int n = 0, pick[MAX_STRONGHOLDS], np = 0;
	if (!children && c.mode == EVAL_B) {
		// Only the count matters, not which ones: count the boxes certainly inside (and already resolved ones)
		// first, then search borderline ones only while they can still decide it.
		int cand[MAX_STRONGHOLDS], nc = 0;
		for (int i = 0; i < c.shRaw; i++) {
			int64_t dmin2, dmax2;
			strongholdBox(c.shRawX[i], c.shRawZ[i], ax, az, dmin2, dmax2);
			if (dmin2 > R2) continue;
			if (dmax2 <= R2) n++;
			else if ((c.shDone >> i) & 1) n += dist2(c.shX[i] * 16 + 4, c.shZ[i] * 16 + 4, ax, az) <= R2;
			else cand[nc++] = i;
		}
		for (int k = 0; k < nc && n < nd.count && n + (nc - k) >= nd.count; k++) {
			int i = cand[k], fx, fz, r = shResolveAny(c, *ring, c.shRawX[i], c.shRawZ[i], c.shFork[i], fx, fz, ax, az, R2);
			if (r == SH_EXACT) {
				c.shX[i] = fx, c.shZ[i] = fz;
				c.shDone |= 1u << i;
				r = dist2(fx * 16 + 4, fz * 16 + 4, ax, az) <= R2 ? SH_IN : SH_OUT;
			}
			n += r == SH_IN;
		}
		if (n < nd.count) return false;
		if (c.hit)  // matched: the report lists the first nd.count in ring order, at exact coordinates
			for (int i = 0, k = 0; i < c.shRaw && k < nd.count; i++) {
				int64_t dmin2, dmax2;
				strongholdBox(c.shRawX[i], c.shRawZ[i], ax, az, dmin2, dmax2);
				if (dmin2 > R2) continue;
				shResolve(c, *ring, i);
				int x = c.shX[i] * 16 + 4, z = c.shZ[i] * 16 + 4;
				if (dmax2 > R2 && dist2(x, z, ax, az) > R2) continue;
				k++;
				addHit(c, WHAT_STRONGHOLD, x, z, depth);
			}
		return true;
	}
	for (int i = 0; i < c.shRaw && n < nd.count; i++) {
		int64_t dmin2, dmax2;
		strongholdBox(c.shRawX[i], c.shRawZ[i], ax, az, dmin2, dmax2);
		if (dmin2 > R2) continue;
		if (c.mode == EVAL_PRE || (!children && dmax2 <= R2)) {
			n++;
			pick[np++] = i;
			continue;
		}
		if (!children && !((c.shDone >> i) & 1)) {
			int fx, fz, r = shResolveAny(c, *ring, c.shRawX[i], c.shRawZ[i], c.shFork[i], fx, fz, ax, az, R2);
			if (r == SH_EXACT) {
				c.shX[i] = fx, c.shZ[i] = fz;
				c.shDone |= 1u << i;
			} else {
				if (r == SH_IN) n++, pick[np++] = i;
				continue;
			}
		} else
			shResolve(c, *ring, i);
		int x = c.shX[i] * 16 + 4, z = c.shZ[i] * 16 + 4;
		if (dist2(x, z, ax, az) > R2) continue;
		if (children) {
			int mark = c.hit ? c.hit->n : 0;
			addHit(c, WHAT_STRONGHOLD, x, z, depth);
			if (evalChildren(c, ni, x, z, 0, depth + 1)) n++;
			else if (c.hit) c.hit->n = mark;
		} else {
			n++;
			pick[np++] = i;
		}
	}
	if (n < nd.count) return false;
	if (c.mode == EVAL_B && c.hit && !children)  // matched: exact coordinates for the report only
		for (int k = 0; k < np; k++) {
			shResolve(c, *ring, pick[k]);
			addHit(c, WHAT_STRONGHOLD, c.shX[pick[k]] * 16 + 4, c.shZ[pick[k]] * 16 + 4, depth);
		}
	return true;
}

// Buildings of the enclosing village (layout from jigsaw.h; exact with c.hg = terrain.h heights, otherwise
// the heights are estimated and the layout is only roughly vanilla's). The layout is cached for the innermost
// enclosing structure. As a prefilter (c.prefilter) the estimated layout needs count - buildingSlack.
HDN bool evalBuilding(EvalCtx& c, const Node& nd, int depth) {
	if (!c.J || !c.pcs || c.curStructure < 0) return false;
	if (c.prefilter && c.q->buildingPass) return true;
	int need = c.prefilter ? nd.count - c.q->buildingSlack : nd.count;
	if (need <= 0) return true;
	bool fresh = c.layoutSt != c.curStructure || c.layoutCx != c.curCx || c.layoutCz != c.curCz;
	if (fresh || c.layoutPartial) {
		int n0 = 0;  // a cached prefix that already holds enough pieces answers without a full layout
		if (!fresh)
			for (int i = 0; i < c.npcs && n0 < need; i++) {
				int cat = pieceCategory(*c.J, c.pcs[i]);
				n0 += cat >= 0 && ((nd.cat[cat >> 6] >> (cat & 63)) & 1);
			}
		if (fresh || n0 < need) {
			// stop the layout once `need` matching pieces are in (the full layout only adds pieces). Exact host check,
			// and the prefilter: there a full layout that overflows (-1) passes too, and loot passes without a layout.
			bool early = c.prefilter || (c.hg && !c.q->hasVillageLoot);
			c.npcs = villageLayout(*c.w, *c.pk, *c.J, c.curStructure, c.curCx, c.curCz, c.pcs, c.maxPcs, c.hg, early ? nd.cat : nullptr, need,
			                       &c.layoutPartial);
			c.layoutSt = c.curStructure, c.layoutCx = c.curCx, c.layoutCz = c.curCz;
		}
	}
	if (c.npcs < 0 && c.prefilter) return true;  // more pieces than the buffer holds: let the exact check decide
	int n = 0;
	for (int i = 0; i < c.npcs && n < need; i++) {
		int cat = pieceCategory(*c.J, c.pcs[i]);
		if (cat < 0 || !((nd.cat[cat >> 6] >> (cat & 63)) & 1)) continue;
		n++;
		int b[6];
		pieceBoxAbs(c.pcs[i], c.curCx, c.curCz, b);
		addHit(c, WHAT_BUILDING0 - cat, (b[0] + b[3]) / 2, (b[2] + b[5]) / 2, depth);
	}
	return n >= need;
}

// Loot of the enclosing structure: at least nd.count of nd.item in its loot containers (loot.h). Buried treasure:
// exact anywhere. Villages: the containers of the layout (exact with c.hg, as buildings); containers that can't
// be predicted (a feature piece draws first in their chunk) never count. A building prefilter passes.
HDN bool evalLoot(EvalCtx& c, const Node& nd, int depth) {
	if (!c.L || c.curStructure < 0) return false;
	int st = c.curStructure;
	if (st == c.L->buriedSt) {
		int n = buriedLoot(*c.L, st, (int64_t)c.seed, c.curCx, c.curCz, nd.item);
		if (n < nd.count) return false;
		addHit(c, WHAT_LOOT0 - (nd.item * 64 + (n < 63 ? n : 63)), c.curCx * 16 + 9, c.curCz * 16 + 9, depth);
		return true;
	}
	if (!c.J || !c.pcs) return false;
	if (c.prefilter) return true;  // decided by the exact check
	if (c.layoutPartial || c.layoutSt != c.curStructure || c.layoutCx != c.curCx || c.layoutCz != c.curCz) {
		c.npcs = villageLayout(*c.w, *c.pk, *c.J, c.curStructure, c.curCx, c.curCz, c.pcs, c.maxPcs, c.hg, nullptr, 0, &c.layoutPartial);
		c.layoutSt = c.curStructure, c.layoutCx = c.curCx, c.layoutCz = c.curCz;
	}
	if (c.npcs <= 0) return false;
	int total = 0;
	EvalCtx* cp = &c;
	const Node* np = &nd;
	forEachChest(*c.L, *c.J, st, c.pcs, c.npcs, c.curCx, c.curCz, (int64_t)c.seed, [&](const LootChest& ch) {
		if (!ch.known || ch.table < 0 || !cp->L->t[ch.table].ok) return;
		int n = lootCount(*cp->L, ch.table, ch.seed, np->item);
		if (n <= 0) return;
		total += n;
		addHit(*cp, WHAT_LOOT0 - (np->item * 64 + (n < 63 ? n : 63)), ch.x, ch.z, depth);
	});
	return total >= nd.count;
}

HDN bool evalNode(EvalCtx& c, int ni, int ax, int az, int slack, int depth) {
	const Node& nd = c.q->n[ni];
	switch (nd.kind) {
	case N_ALL:
		return evalChildren(c, ni, ax, az, slack, depth);
	case N_ANY: {
		bool any = false;
		for (int k = nd.first; k >= 0 && !any; k = c.q->n[k].next) {
			int mark = c.hit ? c.hit->n : 0;
			any = evalNode(c, k, ax, az, slack, depth);
			if (!any && c.hit) c.hit->n = mark;
		}
		return any || nd.first < 0;
	}
	case N_NOT: {
		if (c.mode != EVAL_B) return true;  // can't prune on a negation before the exact check
		if (c.prefilter && hasBuildingBelow(c.q, ni)) return true;  // nor on a building prefilter
		Hit* h = c.hit;
		c.hit = nullptr;  // what a NOT found isn't a match
		bool inner = evalChildren(c, ni, ax, az, slack, depth);
		c.hit = h;
		return !inner;
	}
	case N_STRUCTURE:
		return evalStructure(c, nd, ni, ax, az, slack, depth);
	case N_BIOME:
		return c.mode != EVAL_B || evalBiome(c, nd, ax, az, depth);
	case N_STRONGHOLD:
		return c.mode == EVAL_A || evalStronghold(c, nd, ni, ax, az, slack, depth);
	case N_SPAWN:
		return c.mode != EVAL_B || dist2(c.spawnX, c.spawnZ, 0, 0) <= (int64_t)nd.radius * nd.radius;
	case N_BUILDING:
		return c.mode != EVAL_B || evalBuilding(c, nd, depth);
	case N_LOOT:
		return c.mode != EVAL_B || evalLoot(c, nd, depth);
	}
	return false;
}

// Stage A: placement-only, depends on the low 48 bits only.
HDN bool stageA(const StructTables& T, const Query& q, uint64_t seed) {
	EvalCtx c{};
	c.T = &T;
	c.q = &q;
	c.seed = seed;
	c.mode = EVAL_A;
	return evalNode(c, q.root, 0, 0, q.anchorSpawn ? q.slack : 0, 0);
}

// Stage B: full check for one 64-bit seed. w must be worldInit'ed for that seed. Building conditions: exact
// with hg (terrain.h heights, host only); prefilter = necessary condition for that exact check (estimated
// layouts with Query.buildingSlack); neither: estimated layouts taken at face value (approximate).
HDN bool stageB(const World& w, const PackDev& pk, const StructTables& T, const Query& q, Hit& hit, const JigsawTables* J = nullptr,
                Piece* pieces = nullptr, const HeightGrid* hg = nullptr, bool prefilter = false, int maxPieces = MAX_PIECES,
                const LootTables* L = nullptr, bool warp = false, int phase = 0) {  // phase 1: PRE only, 2: skip PRE (passed)
	EvalCtx c{};
	c.warp = warp;
	c.w = &w;
	c.pk = &pk;
	c.T = &T;
	c.q = &q;
	c.seed = (uint64_t)w.seed;
	c.curStructure = -1;
	c.shRaw = -1;
	c.J = J;
	c.pcs = pieces;
	c.layoutSt = -1;
	c.hg = hg;
	c.maxPcs = maxPieces;
	c.L = L;
	c.prefilter = prefilter && q.hasBuildings;
	hit.seed = w.seed;
	hit.n = 0;
	hit.spawnX = hit.spawnZ = 0;
	// cheap necessary pass first (structures with biome checks, around 0,0 with slack), then the spawn
	c.mode = EVAL_PRE;
	if (!q.exact && phase != 2 && !evalNode(c, q.root, 0, 0, q.anchorSpawn ? q.slack : 0, 0)) return false;
	if (phase == 1) return true;
	int ax = 0, az = 0;
	if (q.anchorSpawn || q.needSpawn) {
#ifdef __CUDA_ARCH__
		if (warp) spawnOriginWarp(w, pk, ax, az);
		else
#endif
			spawnOrigin(w, pk, ax, az);
	}
	hit.spawnX = c.spawnX = ax;
	hit.spawnZ = c.spawnZ = az;
	if (!q.anchorSpawn) ax = az = 0;
	c.mode = EVAL_B;
	c.hit = &hit;
	return evalNode(c, q.root, ax, az, 0, 0);
}

}  // namespace sw

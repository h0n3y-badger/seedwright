// Seed filters as a small condition tree ("2 plains villages within 500 { 1 weaponsmith }, biome
// cherry_grove within 1500, not monument within 800"). Device-capable; nodes live in a flat array.
#pragma once
#include "structures.h"
#include "jigsaw.h"

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
};

enum { MAX_NODES = 48, MAX_HITS = 32, MAX_DEPTH = 6, Y_ANY = -100000, MAX_STRONGHOLDS = 32, MAX_PIECES = 320, MAX_BIOME_LEAVES = 4096 };

struct Node {
	int kind, count, radius, y;
	int first, next;      // first child, next sibling (-1 = none)
	uint64_t structMask;  // N_STRUCTURE: structure indices (bit i = structure i)
	uint32_t setMask;     // sets that can produce one of those structures
	int biome;            // N_BIOME
	int leaf0, nleaf;     // N_BIOME: RTree leaves of that biome in Query::leaves (nleaf 0 = no list, always full search)
	int64_t bbLo[7], bbHi[7];  // N_BIOME: union box of those leaves
	uint64_t cat[3];      // N_BUILDING: jigsaw building categories that count (aliases: weaponsmith|weapon_smith)
};

struct Query {
	int nnodes, root, anchorSpawn, slack, needSpawn;
	int hasBuildings;  // needs village layouts (jigsaw tables + a piece buffer)
	int exact;  // skip the slack-based prefilter (checking given seeds: the prefilter assumes spawn within slack of 0,0)
	// building prefilter (stageB with prefilter=true, estimated terrain; survivors are re-checked with exact heights):
	int buildingSlack;  // estimated layout needs count - buildingSlack buildings (estimates miss real ones)
	int buildingPass;   // 1: building conditions don't filter at all there (exhaustive: every candidate is laid out exactly)
	Node n[MAX_NODES];
	int nleaves;
	uint16_t leaves[MAX_BIOME_LEAVES];
};

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
enum { WHAT_STRONGHOLD = -1000, WHAT_SPAWN = -1001, WHAT_BUILDING0 = -2000 };  // biomes: -1 - biome

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
	const HeightGrid* hg;  // exact terrain heights (host, terrain.h); null: estimateSurfaceY
	int prefilter;         // building conditions are a necessary-condition prefilter (see Query.buildingSlack)
	int maxPcs;            // size of pcs
};

HDN bool hasBuildingBelow(const Query* q, int ni) {
	for (int k = q->n[ni].first; k >= 0; k = q->n[k].next)
		if (q->n[k].kind == N_BUILDING || hasBuildingBelow(q, k)) return true;
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
	if (nd.nleaf > 0 && wit >= 0 && pk.rnodes[wit].biome != nd.biome) {
		int64_t dW = nodeDist(pk.rnodes[wit], t), lb = 0;
		for (int k = 0; k < 7; k++) {
			int64_t d = pdist(nd.bbLo[k], nd.bbHi[k], t[k]);
			lb += d * d;
		}
		if (lb > dW) return false;
		if (nd.nleaf <= 128) {  // per-leaf check; for big biomes (plains, desert) the full search is cheaper
			bool reach = false;
			for (int i = 0; i < nd.nleaf && !reach; i++) reach = nodeDist(pk.rnodes[q.leaves[nd.leaf0 + i]], t) <= dW;
			if (!reach) return false;
		}
	}
	wit = rtreeSearch(pk, 0, t, -1);
	return pk.rnodes[wit].biome == nd.biome;
}

HDN bool evalBiome(EvalCtx& c, const Node& nd, int ax, int az, int depth) {
	// on a 16-block grid, nearest rings first; y=any tries the cave levels (y -48..48, every 16).
	// Same samples and order as biomeAt(..., 0); y-independent climate ops are evaluated once per column.
	int64_t R2 = (int64_t)nd.radius * nd.radius;
	int steps = nd.radius / 16;
	int y0 = nd.y == Y_ANY ? -48 : nd.y, y1 = nd.y == Y_ANY ? 48 : nd.y;
	const PackDev& pk = *c.pk;
	int upto = climateUpto(pk), wit = -1;
	double pv[MAX_OPS], vv[MAX_OPS];
	for (int r = 0; r <= steps; r++)
		for (int dz = -r; dz <= r; dz++)
			for (int dx = -r; dx <= r; dx += (dz == -r || dz == r) ? 1 : 2 * r) {  // ring cells only
				int bx = ax + dx * 16, bz = az + dz * 16;
				if (dist2(bx, bz, ax, az) > R2) continue;
				for (int y = y0; y <= y1; y += 16) {
					evalOpsState(*c.w, pk, (bx >> 2) << 2, (y >> 2) << 2, (bz >> 2) << 2, false, false, upto, pv, vv, y != y0);
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

// Strongholds within radius (+slack) of the anchor. A stronghold ends up within 112 blocks (+ chunk
// rounding) of its raw ring position, so only those whose raw position could be in range get the
// expensive biome search; PRE mode decides on raw positions alone.
HDN void shResolve(EvalCtx& c, const SetDev& ring, int i) {
	if ((c.shDone >> i) & 1) return;
	strongholdResolve(*c.w, *c.pk, ring, c.shRawX[i], c.shRawZ[i], c.shFork[i], c.shX[i], c.shZ[i]);
	c.shDone |= 1u << i;
}

HDN bool evalStronghold(EvalCtx& c, const Node& nd, int ni, int ax, int az, int slack, int depth) {
	const SetDev* ring = ringSet(*c.T);
	if (!ring) return false;
	if (c.shRaw < 0) c.shRaw = strongholdRings(c.seed, *ring, MAX_STRONGHOLDS, c.shRawX, c.shRawZ, c.shFork);
	const int margin = 112 + 24;  // final position vs raw ring chunk centre
	int64_t R = nd.radius + slack;
	int64_t Rout = R + margin, Rin = R - margin, R2 = R * R;
	bool children = nd.first >= 0;
	// decide without biome searches where the bound settles it: certainly in (raw well inside), certainly
	// out (raw well outside); only borderline ones are resolved. Children need exact positions.
	int n = 0, pick[MAX_STRONGHOLDS], np = 0;
	for (int i = 0; i < c.shRaw && n < nd.count; i++) {
		int64_t d2 = dist2(c.shRawX[i] * 16 + 8, c.shRawZ[i] * 16 + 8, ax, az);
		if (d2 > Rout * Rout) continue;
		if (c.mode == EVAL_PRE || (!children && Rin > 0 && d2 <= Rin * Rin)) {
			n++;
			pick[np++] = i;
			continue;
		}
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
	if (c.layoutSt != c.curStructure || c.layoutCx != c.curCx || c.layoutCz != c.curCz) {
		c.npcs = villageLayout(*c.w, *c.pk, *c.J, c.curStructure, c.curCx, c.curCz, c.pcs, c.maxPcs, c.hg);
		c.layoutSt = c.curStructure, c.layoutCx = c.curCx, c.layoutCz = c.curCz;
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
                Piece* pieces = nullptr, const HeightGrid* hg = nullptr, bool prefilter = false, int maxPieces = MAX_PIECES) {
	EvalCtx c{};
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
	c.prefilter = prefilter && q.hasBuildings;
	hit.seed = w.seed;
	hit.n = 0;
	hit.spawnX = hit.spawnZ = 0;
	// cheap necessary pass first (structures with biome checks, around 0,0 with slack), then the spawn
	c.mode = EVAL_PRE;
	if (!q.exact && !evalNode(c, q.root, 0, 0, q.anchorSpawn ? q.slack : 0, 0)) return false;
	int ax = 0, az = 0;
	if (q.anchorSpawn || q.needSpawn) spawnOrigin(w, pk, ax, az);
	hit.spawnX = c.spawnX = ax;
	hit.spawnZ = c.spawnZ = az;
	if (!q.anchorSpawn) ax = az = 0;
	c.mode = EVAL_B;
	c.hit = &hit;
	return evalNode(c, q.root, ax, az, 0, 0);
}

}  // namespace sw

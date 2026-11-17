// Jigsaw structure assembly (vanilla JigsawPlacement), host + device. Written from vanilla's own sources:
// JigsawStructure.findGenerationPoint, JigsawPlacement.addPieces / Placer.tryPlacingChildren, the pool element
// classes, StructureTemplate (jigsaw order, rotation transform, bounding boxes), Util.shuffle,
// SequencedPriorityIterator and the VoxelShape free-space bookkeeping.
//
// Everything is exact except the terrain height (ChunkGenerator.getFirstFreeHeight, WORLD_SURFACE_WG), which
// is estimated by estimateSurfaceY() until seedwright has real terrain; a HeightGrid can supply true heights
// (the oracle dumps them) and then the layout is vanilla's. Heights enter at the start piece (heightmap
// projection), at terrain_matching connections, and therefore in collisions/vertical bounds; the RNG call
// sequence only depends on them through which pieces fit.
//
// Pool aliases (trial chambers) are not supported (JStruct.supported = 0).
//
// Use (host): JigsawHost H = buildJigsaw(pack);  H.t is a POD JigsawTables (~280 KB, cudaMemcpy it once).
//   int cat = H.categoryIndex("weaponsmith");          // building category, see jigsawCategories(H, st)
// per candidate chunk of a village set (host or device):
//   int st = jigsawStructureAt(w, pk, T, J, set, cx, cz);    // like structureAt, start checked where vanilla does
//   Piece pc[320]; int n = villageLayout(w, pk, J, st, cx, cz, pc, 320);
//   if (n > 0 && countCategory(J, pc, n, cat) >= 2) ...
#pragma once
#include "structures.h"
#include "pack.h"
#include <algorithm>
#include <functional>

namespace sw {

enum {
	JMAX_TEMPLATES = 2048, JMAX_JIGSAWS = 8192, JMAX_ELEMENTS = 4096, JMAX_SUBS = 256, JMAX_POOLS = 512, JMAX_ENTRIES = 8192,
	JMAX_STRUCTS = 48, JMAX_CATS = 160,
	JMAX_SHUFFLE = 256,  // pool + fallback expanded (weighted) entries per connection (villages: <= 91)
	JMAX_PJIG = 64,  // jigsaw blocks per template
};

enum JKind : uint8_t { JK_SINGLE, JK_FEATURE, JK_LIST, JK_EMPTY };
enum JHeight : uint8_t { JH_CONST, JH_UNIFORM, JH_UNSUPPORTED };

struct JTpl {
	int16_t sx, sy, sz, cat;  // cat: building category (-1 = not a building)
	uint16_t j0, nj;
};
struct JJig {
	int16_t x, y, z;
	uint8_t front, top, rollable;
	int16_t name, target, pool;  // pool: -1 = none / non-existent / minecraft:empty (no children, no RNG)
	int16_t placement, selection;
};
struct JElem {
	uint8_t kind, rigid;
	int16_t tpl;  // single/legacy: template; list: template of the first member (or -1)
	uint16_t sub0, nsub;  // list members
	int16_t expand[4];  // expansion-hack height per rotation (0 = none), already includes the ySpan <= 16 test
};
struct JPool {
	uint16_t e0, n;  // expanded (weight-repeated) element list in ent[]
	int16_t fallback, maxSize;
	uint8_t ok;  // vanilla uses it (exists, non-empty or minecraft:empty, same for its fallback)
};
struct JStruct {
	int16_t startPool, startJigsaw, maxDepth, maxH, maxV, padBottom, padTop;
	uint8_t heightKind, project, expansion, supported;
	int h0, h1;
};

struct JigsawTables {
	int ntpl, njig, nelem, nsub, npool, nent, nstruct, ncat;
	int featureName;  // name of FeaturePoolElement's jigsaw (-1 = null: matches any target, 26.3+)
	int structOf[MAX_STRUCTS];  // pack structure index -> jigsaw structure (-1 = not a jigsaw structure)
	JTpl tpl[JMAX_TEMPLATES];
	JJig jig[JMAX_JIGSAWS];
	JElem elem[JMAX_ELEMENTS];
	uint16_t sub[JMAX_SUBS];
	JPool pool[JMAX_POOLS];
	uint16_t ent[JMAX_ENTRIES];
	JStruct st[JMAX_STRUCTS];
};

// One placed piece. Coordinates are relative to the start chunk's min corner (chunkX*16, 0, chunkZ*16):
// add those back for world coordinates (pieceBoxAbs). Bounding box is inclusive (vanilla BoundingBox).
struct Piece {
	int16_t elem;  // JigsawTables.elem index; JElem.tpl gives the template
	uint8_t rot, depth;  // Rotation ordinal (NONE, CLOCKWISE_90, CLOCKWISE_180, COUNTERCLOCKWISE_90), jigsaw depth
	int16_t x0, y0, z0, x1, y1, z1;
	int16_t region;  // free-space shape the piece was carved from: 0 = the structure's, k = inside piece k-1
	int16_t prio;  // placement priority of the connection that queued it (-32768 = not queued / done)
};

static_assert(sizeof(Piece) == 20, "Piece layout");

struct HeightGrid {  // optional true heights (first free y, WORLD_SURFACE_WG), absolute coordinates
	const int16_t* h;
	int x0, z0, nx, nz;
	// host only: exact height for columns outside the grid (terrain.h TerrainWorld), e.g. heightFn = terrainHeightFn
	int (*fn)(void* ctx, int x, int z) = nullptr;
	void* ctx = nullptr;
};
enum { HEIGHT_UNKNOWN = -32768 };

// ---------------------------------------------------------------- geometry

HD void dirVec(int d, int& x, int& y, int& z) {  // Direction: down up north south west east
	const int8_t v[6][3] = {{0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}, {-1, 0, 0}, {1, 0, 0}};
	x = v[d][0];
	y = v[d][1];
	z = v[d][2];
}
// StructureTemplate.transform with pivot 0, no mirror
HD void rotXZ(int rot, int x, int z, int& ox, int& oz) {
	switch (rot) {
	case 1: ox = -z; oz = x; break;  // CLOCKWISE_90
	case 2: ox = -x; oz = -z; break;  // CLOCKWISE_180
	case 3: ox = z; oz = -x; break;  // COUNTERCLOCKWISE_90
	default: ox = x; oz = z;
	}
}
HD int rotDir(int rot, int d) {
	if (d < 2) return d;
	int x, y, z, rx, rz;
	dirVec(d, x, y, z);
	rotXZ(rot, x, z, rx, rz);
	return rx < 0 ? 4 : rx > 0 ? 5 : rz < 0 ? 2 : 3;
}

struct JBox {
	int x0, y0, z0, x1, y1, z1;
};

// element bounding box at template origin (px,py,pz) with rotation (SinglePoolElement / FeaturePoolElement / ListPoolElement)
HDN JBox elemBox(const JigsawTables& J, int e, int px, int py, int pz, int rot) {
	const JElem& E = J.elem[e];
	JBox b;
	if (E.kind == JK_SINGLE) {
		const JTpl& T = J.tpl[E.tpl];
		int ax, az;
		rotXZ(rot, T.sx - 1, T.sz - 1, ax, az);
		int ay = T.sy - 1;
		b.x0 = px + (ax < 0 ? ax : 0);
		b.x1 = px + (ax > 0 ? ax : 0);
		b.y0 = py + (ay < 0 ? ay : 0);
		b.y1 = py + (ay > 0 ? ay : 0);
		b.z0 = pz + (az < 0 ? az : 0);
		b.z1 = pz + (az > 0 ? az : 0);
		return b;
	}
	if (E.kind == JK_LIST) {
		bool any = false;
		for (int i = 0; i < E.nsub; i++) {
			int s = J.sub[E.sub0 + i];
			if (J.elem[s].kind == JK_EMPTY) continue;
			JBox c = elemBox(J, s, px, py, pz, rot);
			if (!any) b = c;
			else {
				b.x0 = c.x0 < b.x0 ? c.x0 : b.x0; b.y0 = c.y0 < b.y0 ? c.y0 : b.y0; b.z0 = c.z0 < b.z0 ? c.z0 : b.z0;
				b.x1 = c.x1 > b.x1 ? c.x1 : b.x1; b.y1 = c.y1 > b.y1 ? c.y1 : b.y1; b.z1 = c.z1 > b.z1 ? c.z1 : b.z1;
			}
			any = true;
		}
		if (any) return b;
	}
	b.x0 = b.x1 = px;  // feature: zero size
	b.y0 = b.y1 = py;
	b.z0 = b.z1 = pz;
	return b;
}

// getShuffledJigsawBlocks: template jigsaws (palette order), Util.shuffle, stable sort by selection priority
// (highest first). Returns count; out[] holds J.jig indices, or -1 for the feature element's synthetic jigsaw.
HDN int shuffledJigsaws(const JigsawTables& J, int e, JavaRandom& r, int* out) {
	const JElem* E = &J.elem[e];
	if (E->kind == JK_LIST) return E->nsub ? shuffledJigsaws(J, J.sub[E->sub0], r, out) : 0;  // the first element's jigsaws
	if (E->kind == JK_FEATURE) {
		out[0] = -1;
		return 1;
	}
	if (E->kind != JK_SINGLE) return 0;
	const JTpl& T = J.tpl[E->tpl];
	int n = T.nj;
	for (int i = 0; i < n; i++) out[i] = T.j0 + i;
	for (int i = n; i > 1; i--) {
		int k = r.nextInt(i);
		int t = out[i - 1];
		out[i - 1] = out[k];
		out[k] = t;
	}
	// stable insertion sort, descending selection priority
	for (int i = 1; i < n; i++) {
		int v = out[i], p = J.jig[v].selection, k = i - 1;
		while (k >= 0 && J.jig[out[k]].selection < p) {
			out[k + 1] = out[k];
			k--;
		}
		out[k + 1] = v;
	}
	return n;
}

// a jigsaw block as placed: world-relative position + rotated orientation
struct JPlaced {
	int x, y, z, front, top, rollable, name, target, pool, placement;
};
HD JPlaced placeJig(const JigsawTables& J, int ji, int px, int py, int pz, int rot) {
	JPlaced p;
	if (ji < 0) {  // FeaturePoolElement: at the origin, front DOWN, top SOUTH, rollable, pool empty, target empty
		p.x = px; p.y = py; p.z = pz;
		p.front = 0; p.top = 3; p.rollable = 1;
		p.name = J.featureName; p.target = -2; p.pool = -1; p.placement = 0;
		return p;
	}
	const JJig& g = J.jig[ji];
	int rx, rz;
	rotXZ(rot, g.x, g.z, rx, rz);
	p.x = px + rx; p.y = py + g.y; p.z = pz + rz;
	p.front = rotDir(rot, g.front);
	p.top = rotDir(rot, g.top);
	p.rollable = g.rollable;
	p.name = g.name; p.target = g.target; p.pool = g.pool; p.placement = g.placement;
	return p;
}

HD int oppositeDir(int d) { return d ^ 1; }  // down<->up, north<->south, west<->east

// JigsawBlock.canAttach (name -1 = null name: 26.3 feature jigsaw, attaches to any target)
HD bool canAttach(const JPlaced& s, const JPlaced& t) {
	return s.front == oppositeDir(t.front) && (s.rollable || s.top == t.top) && (t.name == -1 || s.target == t.name);
}

// ---------------------------------------------------------------- terrain height

// Estimated ChunkGenerator.getFirstFreeHeight(x, z, WORLD_SURFACE_WG): one above the highest non-air block.
// Uses the router's `depth` (y gradient + terrain offset; linear in y over the world range): the surface
// estimate is where depth crosses 0, i.e. the same "depth >= 0" surface structures.h uses, at block
// resolution. Water fills up to sea level - 1, so the result is never below the sea level.
HDN int estimateSurfaceY(const World& w, const PackDev& pk, int x, int z) {
	double pv[MAX_OPS], vv[MAX_OPS];
	int d = pk.climate[4];
	const int ya = 0, yb = 160;
	evalOpsState(w, pk, x, ya, z, false, false, d, pv, vv, false);
	double a = pv[d];
	evalOpsState(w, pk, x, yb, z, false, false, d, pv, vv, true);
	double b = pv[d];
	int top;
	if (a == b) top = a >= 0.0 ? pk.minY + pk.height - 1 : pk.minY;
	else {
		double y = ya + a * (double)(yb - ya) / (a - b);  // depth(y) = 0
		top = (int)floor(y);
		int lo = pk.minY, hi = pk.minY + pk.height - 1;
		top = top < lo ? lo : top > hi ? hi : top;
	}
	int h = top + 1;
	return h < pk.seaLevel ? pk.seaLevel : h;
}

HDN int jigsawHeight(const World& w, const PackDev& pk, const HeightGrid* g, int x, int z) {
	if (g && g->h) {
		int i = x - g->x0, k = z - g->z0;
		if (i >= 0 && k >= 0 && i < g->nx && k < g->nz) {
			int h = g->h[k * g->nx + i];
			if (h != HEIGHT_UNKNOWN) return h;
		}
	}
#ifndef __CUDA_ARCH__
	if (g && g->fn) return g->fn(g->ctx, x, z);
#endif
	return estimateSurfaceY(w, pk, x, z);
}

// ---------------------------------------------------------------- start piece

struct JStart {
	int elem, rot;
	JBox box;  // after the heightmap projection (absolute coordinates)
	int px, py, pz;  // template origin (absolute)
	int centerX, centerY, centerZ;  // GenerationStub position (biome check)
};

// JigsawStructure.findGenerationPoint + the first half of JigsawPlacement.addPieces. r must be the structure
// generation random (WorldgenRandom.setLargeFeatureSeed(seed, chunkX, chunkZ)). false = no structure.
HDN bool jigsawStart(const World& w, const PackDev& pk, const JigsawTables& J, int js, int cx, int cz, JavaRandom& r, const HeightGrid* g,
                     JStart& s) {
	const JStruct& S = J.st[js];
	if (!S.supported) return false;
	int y = S.h0;
	if (S.heightKind == JH_UNIFORM) y = S.h1 < S.h0 ? S.h0 : r.nextInt(S.h1 - S.h0 + 1) + S.h0;  // Mth.randomBetweenInclusive
	int x = cx << 4, z = cz << 4;
	s.rot = r.nextInt(4);  // Rotation.getRandom
	const JPool& P = J.pool[S.startPool];
	if (P.n == 0) return false;
	s.elem = J.ent[P.e0 + r.nextInt(P.n)];  // getRandomTemplate
	if (J.elem[s.elem].kind == JK_EMPTY) return false;
	int ax = 0, ay = 0, az = 0;  // local anchor (start_jigsaw_name)
	if (S.startJigsaw >= 0) {
		int lst[JMAX_PJIG];
		int n = shuffledJigsaws(J, s.elem, r, lst), i = 0;
		for (; i < n; i++) {
			JPlaced p = placeJig(J, lst[i], x, y, z, s.rot);
			if (p.name == S.startJigsaw) {
				ax = p.x - x;
				ay = p.y - y;
				az = p.z - z;
				break;
			}
		}
		if (i == n) return false;
	}
	s.px = x - ax;
	s.py = y - ay;
	s.pz = z - az;
	s.box = elemBox(J, s.elem, s.px, s.py, s.pz, s.rot);
	s.centerX = (s.box.x1 + s.box.x0) / 2;  // Java int division (truncates toward zero)
	s.centerZ = (s.box.z1 + s.box.z0) / 2;
	int bottomY = S.project ? y + jigsawHeight(w, pk, g, s.centerX, s.centerZ) : s.py;
	int dy = bottomY - (s.box.y0 + 1);  // getGroundLevelDelta() == 1 for every element type
	s.box.y0 += dy;
	s.box.y1 += dy;
	s.py += dy;
	s.centerY = bottomY + ay;
	return true;
}

// Structure.findValidGenerationPoint's biome test for a jigsaw start: 26.3 also requires a valid biome
// somewhere in the column at the start piece's center (JigsawPlacement, projected starts only); then the
// biome at the GenerationStub position (centerY uses the height estimate unless a grid is given).
HDN bool jigsawBiomeOk(const World& w, const PackDev& pk, const StructTables& T, int st, const JStart& s, bool projected) {
	const StructDev& D = T.st[st];
	if (!maskHas(D.biomes, biomeAt(w, pk, s.centerX >> 2, s.centerY >> 2, s.centerZ >> 2, 2))) return false;
	if (pk.legacy || !projected) return true;
	Column col;
	columnClimate(w, pk, s.centerX >> 2, s.centerZ >> 2, col);
	int leaf = -1, q0 = pk.minY >> 2, q1 = (pk.minY + pk.height - 1) >> 2;  // couldStructureExistInColumn(minY, maxY)
	for (int qy = q0; qy <= q1; qy++)
		if (maskHas(D.biomes, columnBiome(pk, col, qy, leaf))) return true;
	return false;
}

// structureAt (weighted retry over the set's structures) with jigsaw starts validated at their real start
// position instead of the chunk middle. Non-jigsaw structures use structureBiomeOk as before.
HDN int jigsawStructureAt(const World& w, const PackDev& pk, const StructTables& T, const JigsawTables& J, int set, int cx, int cz,
                          const HeightGrid* g = nullptr) {
	const SetDev& s = T.sets[set];
	JavaRandom r;
	largeFeatureRandom(r, (uint64_t)w.seed, cx, cz);
	int alive[MAX_SET_STRUCTS], n = s.nstructs, total = 0;
	for (int i = 0; i < n; i++) {
		alive[i] = i;
		total += s.weights[i];
	}
	while (n > 0) {
		int pick;
		if (s.nstructs == 1) pick = 0;
		else {
			int choice = r.nextInt(total), idx = 0;
			for (; idx < n; idx++) {
				choice -= s.weights[alive[idx]];
				if (choice < 0) break;
			}
			pick = alive[idx];
			total -= s.weights[pick];
			for (int j = idx; j < n - 1; j++) alive[j] = alive[j + 1];
		}
		n--;
		int st = s.structs[pick], js = J.structOf[st];
		bool ok;
		if (js >= 0 && J.st[js].supported) {
			JavaRandom gr;
			largeFeatureRandom(gr, (uint64_t)w.seed, cx, cz);
			JStart start;
			ok = jigsawStart(w, pk, J, js, cx, cz, gr, g, start) && jigsawBiomeOk(w, pk, T, st, start, J.st[js].project);
		} else
			ok = structureBiomeOk(w, pk, T, st, cx, cz);
		if (ok) return st;
		if (s.nstructs == 1) break;
	}
	return -1;
}

// ---------------------------------------------------------------- layout

// template origin (PoolElementStructurePiece.position) of a piece: its box min corner minus the element's box
// min corner at origin 0 (the expansion hack only raises maxY)
HDN void pieceOrigin(const JigsawTables& J, const Piece& p, int& px, int& py, int& pz) {
	JBox b0 = elemBox(J, p.elem, 0, 0, 0, p.rot);
	px = p.x0 - b0.x0;
	py = p.y0 - b0.y0;
	pz = p.z0 - b0.z0;
}

// RNG advance of Util.shuffle on n items without computing it: java.util.Random.nextInt(i) for i = n..2
// (powers of two: one step; otherwise one step unless rejected: bits >= floor(2^31 / i) * i).
HD void skipShuffle(JavaRandom& r, int n) {
	for (int i = n; i > 1; i--) {
		if ((i & -i) == i) {
			r.next(31);
			continue;
		}
		uint32_t lim = (0x80000000u / (uint32_t)i) * (uint32_t)i;
		while ((uint32_t)r.next(31) >= lim) {
		}
	}
}

// the element whose jigsaw list an element uses (list: first member): its first J.jig index (-1 = feature's
// synthetic jigsaw) and the count
HDN int jigsawSource(const JigsawTables& J, int e, int& base0) {
	const JElem& E = J.elem[e];
	if (E.kind == JK_LIST) {
		if (!E.nsub) {
			base0 = 0;
			return 0;
		}
		return jigsawSource(J, J.sub[E.sub0], base0);
	}
	if (E.kind == JK_FEATURE) {
		base0 = -1;
		return 1;
	}
	if (E.kind != JK_SINGLE) {
		base0 = 0;
		return 0;
	}
	base0 = J.tpl[E.tpl].j0;
	return J.tpl[E.tpl].nj;
}

// Placer: target box for connecting target jigsaw tj (at rotation tr, positioned at ZERO) to source jigsaw sj
HDN void connectBox(const World& w, const PackDev& pk, const HeightGrid* g, const JigsawTables& J, int te, int tr, const JPlaced& tj, const JPlaced& sj,
                    int tx, int ty, int tz, int fy, int srcLocalY, bool bothRigid, int srcBoxY, int expandTo, int ox, int oz, int& baseH, JBox& bb,
                    int& rpy) {
	int rpx = tx - tj.x, rpz = tz - tj.z;
	rpy = ty - tj.y;
	bb = elemBox(J, te, rpx, rpy, rpz, tr);
	int deltaY = srcLocalY - tj.y + fy;
	int tBoxY;
	if (bothRigid) tBoxY = srcBoxY + deltaY;
	else {
		if (baseH == HEIGHT_UNKNOWN) baseH = jigsawHeight(w, pk, g, sj.x + ox, sj.z + oz);
		tBoxY = baseH - tj.y;
	}
	int yo = tBoxY - bb.y0;
	bb.y0 += yo;
	bb.y1 += yo;
	rpy += yo;
	if (expandTo > 0) {
		int ns = expandTo + 1 > bb.y1 - bb.y0 ? expandTo + 1 : bb.y1 - bb.y0;
		if (bb.y0 + ns > bb.y1) bb.y1 = bb.y0 + ns;
	}
}

// Fits = vanilla's !joinIsNotEmpty(free, box.deflate(0.25), ONLY_SECOND): every block of the box lies in the
// free shape. Shape `region` is its base box minus every piece carved from it (pieces with that region).
HDN bool jFits(const Piece* pc, int n, const JBox& base, int region, const JBox& b) {
	if (b.x0 < base.x0 || b.x1 > base.x1 || b.y0 < base.y0 || b.y1 > base.y1 || b.z0 < base.z0 || b.z1 > base.z1) return false;
	for (int i = 0; i < n; i++) {
		const Piece& p = pc[i];
		if (p.region != region) continue;
		if (b.x0 <= p.x1 && p.x0 <= b.x1 && b.z0 <= p.z1 && p.z0 <= b.z1 && b.y0 <= p.y1 && p.y0 <= b.y1) return false;
	}
	return true;
}

// Assemble the jigsaw structure `structureIndex` (pack structure index) started at chunk (chunkX, chunkZ).
// Returns the number of pieces written to out (vanilla order: start piece first, then placement order),
// 0 = no structure (empty start element / unsupported), -1 = more than maxOut pieces.
// Does not re-check the biome: pick the structure with jigsawStructureAt / structureAt first.
// Device stack use: ~2.5 KB (+ ~2 KB for the height estimate) + the caller's maxOut * sizeof(Piece) (20 B); villages reach ~280 pieces.
HD int pieceCategory(const JigsawTables& J, const Piece& p);
// stopCat/stopNeed: stop early (*partial = true) once stopNeed pieces of a category in stopCat (Node.cat bit mask)
// are placed. Pieces are only ever appended, so the partial layout is a prefix of the full one.
HDN int jigsawLayout(const World& w, const PackDev& pk, const JigsawTables& J, int structureIndex, int chunkX, int chunkZ, Piece* out, int maxOut,
                     const HeightGrid* g = nullptr, const uint64_t* stopCat = nullptr, int stopNeed = 0, bool* partial = nullptr) {
	if (partial) *partial = false;
	int stopHave = 0;
	int js = structureIndex >= 0 && structureIndex < MAX_STRUCTS ? J.structOf[structureIndex] : -1;
	if (js < 0 || maxOut < 1) return 0;
	const JStruct& S = J.st[js];
	JavaRandom r;
	largeFeatureRandom(r, (uint64_t)w.seed, chunkX, chunkZ);  // Structure.GenerationContext.makeRandom
	JStart s;
	if (!jigsawStart(w, pk, J, js, chunkX, chunkZ, r, g, s)) return 0;
	const int ox = chunkX << 4, oz = chunkZ << 4;  // pieces are stored relative to this
	int n = 1;
	{
		Piece& c = out[0];
		c.elem = (int16_t)s.elem;
		c.rot = (uint8_t)s.rot;
		c.depth = 0;
		c.x0 = (int16_t)(s.box.x0 - ox); c.y0 = (int16_t)s.box.y0; c.z0 = (int16_t)(s.box.z0 - oz);
		c.x1 = (int16_t)(s.box.x1 - ox); c.y1 = (int16_t)s.box.y1; c.z1 = (int16_t)(s.box.z1 - oz);
		c.region = 0;  // the AABB shape is built minus the start box
		c.prio = -32768;
	}
	if (S.maxDepth <= 0) return 1;
	if (stopCat) {
		int cat = pieceCategory(J, out[0]);
		if (cat >= 0 && ((stopCat[cat >> 6] >> (cat & 63)) & 1) && ++stopHave >= stopNeed) {
			*partial = true;
			return 1;
		}
	}
	JBox aabb;  // AABB(centerX - h, max(centerY - v, minY + padBottom), ..., centerX + h + 1, min(centerY + v + 1, maxY + 1 - padTop), ...)
	{
		int cxr = s.centerX - ox, czr = s.centerZ - oz;
		int ylo = s.centerY - S.maxV, ymin = pk.minY + S.padBottom;
		int yhi = s.centerY + S.maxV + 1, ymax = pk.minY + pk.height - S.padTop;
		aabb.x0 = cxr - S.maxH; aabb.x1 = cxr + S.maxH;  // inclusive block ranges
		aabb.z0 = czr - S.maxH; aabb.z1 = czr + S.maxH;
		aabb.y0 = ylo > ymin ? ylo : ymin;
		aabb.y1 = (yhi < ymax ? yhi : ymax) - 1;
	}
	out[0].prio = 0x7fff;  // the start piece is processed first, outside the queue
	int srcJ[JMAX_PJIG], tgtJ[JMAX_PJIG];
	uint16_t list[JMAX_SHUFFLE];
	for (;;) {
		// SequencedPriorityIterator: highest priority first, FIFO (= piece order) within a priority
		int si = -1, best = -32768;
		for (int i = 0; i < n; i++)
			if (out[i].prio != -32768 && (si < 0 || out[i].prio > best)) {
				si = i;
				best = out[i].prio;
			}
		if (si < 0) break;
		out[si].prio = -32768;
		const Piece src = out[si];
		int depth = src.depth;
		// context shape: the shape the source was carved from (start piece: the structure shape)
		int ctxRegion = src.region;
		const JElem& SE = J.elem[src.elem];
		bool srcRigid = SE.rigid;
		int nsj = shuffledJigsaws(J, src.elem, r, srcJ);
		int spx, spy, spz;
		pieceOrigin(J, src, spx, spy, spz);
		for (int a = 0; a < nsj; a++) {
			JPlaced sj = placeJig(J, srcJ[a], spx, spy, spz, src.rot);
			int fx, fy, fz;
			dirVec(sj.front, fx, fy, fz);
			int tx = sj.x + fx, ty = sj.y + fy, tz = sj.z + fz;
			int srcLocalY = sj.y - src.y0;
			int baseH = HEIGHT_UNKNOWN;
			if (sj.pool < 0 || !J.pool[sj.pool].ok) continue;  // none / "Empty or non-existent pool"
			const JPool& TP = J.pool[sj.pool];
			const JPool& FP = J.pool[TP.fallback];
			bool inside = tx >= src.x0 && tx <= src.x1 && ty >= src.y0 && ty <= src.y1 && tz >= src.z0 && tz <= src.z1;
			int region = inside ? si + 1 : ctxRegion;
			JBox base;
			if (region == 0) base = aabb;
			else {
				const Piece& b = out[region - 1];
				base.x0 = b.x0; base.y0 = b.y0; base.z0 = b.z0; base.x1 = b.x1; base.y1 = b.y1; base.z1 = b.z1;
			}
			// target candidates: shuffled pool (unless at max depth) + shuffled fallback
			int nl = 0;
			if (depth != S.maxDepth) {
				for (int i = 0; i < TP.n; i++) list[nl + i] = J.ent[TP.e0 + i];
				for (int i = TP.n; i > 1; i--) {
					int k = r.nextInt(i);
					uint16_t t = list[nl + i - 1];
					list[nl + i - 1] = list[nl + k];
					list[nl + k] = t;
				}
				nl += TP.n;
			}
			for (int i = 0; i < FP.n; i++) list[nl + i] = J.ent[FP.e0 + i];
			for (int i = FP.n; i > 1; i--) {
				int k = r.nextInt(i);
				uint16_t t = list[nl + i - 1];
				list[nl + i - 1] = list[nl + k];
				list[nl + k] = t;
			}
			nl += FP.n;
			// Vanilla tries each candidate element in 4 shuffled rotations, each with its shuffled jigsaw list, and
			// takes the first (rotation, jigsaw) that attaches and fits. Which ones fit doesn't depend on the order,
			// so fits are worked out first and the shuffles are only materialized where the order decides the
			// result; elsewhere the RNG is just advanced by the same calls (skipShuffle). Elements that failed
			// once at this connection fail again (the free shape only changes on success).
			int failed[32], nfailed = 0;
			bool placed = false;
			for (int li = 0; li < nl && !placed; li++) {
				int te = list[li];
				const JElem& TE = J.elem[te];
				if (TE.kind == JK_EMPTY) break;
				int base0, cnt = jigsawSource(J, te, base0);
				bool known = false;
				for (int k = 0; k < nfailed && !known; k++) known = failed[k] == te;
				uint64_t mask[4] = {0, 0, 0, 0};
				if (!known) {
					bool tgtRigid = TE.rigid;
					int expandTo0 = S.expansion;
					for (int tr = 0; tr < 4; tr++)
						for (int k = 0; k < cnt; k++) {
							JPlaced tj = placeJig(J, base0 < 0 ? -1 : base0 + k, 0, 0, 0, tr);
							if (!canAttach(sj, tj)) continue;
							JBox bb;
							int rpy;
							if (baseH == HEIGHT_UNKNOWN && !(srcRigid && tgtRigid)) {
								// x/z don't depend on the height: a box outside the free shape's base in x/z can't fit at any y,
								// so don't sample the terrain column for it (only the y placement needs baseH)
								JBox xz = elemBox(J, te, tx - tj.x, 0, tz - tj.z, tr);
								if (xz.x0 < base.x0 || xz.x1 > base.x1 || xz.z0 < base.z0 || xz.z1 > base.z1) continue;
							}
							connectBox(w, pk, g, J, te, tr, tj, sj, tx, ty, tz, fy, srcLocalY, srcRigid && tgtRigid, src.y0, expandTo0 ? TE.expand[tr] : 0, ox,
							           oz, baseH, bb, rpy);
							if (jFits(out, n, base, region, bb)) mask[tr] |= 1ULL << k;
						}
				}
				if (known || (mask[0] | mask[1] | mask[2] | mask[3]) == 0) {
					skipShuffle(r, 4);  // Rotation.getShuffled
					for (int i = 0; i < 4; i++) skipShuffle(r, base0 < 0 ? 0 : cnt);  // per rotation: getShuffledJigsawBlocks
					if (!known && nfailed < 32) failed[nfailed++] = te;
					continue;
				}
				int rots[4] = {0, 1, 2, 3};  // Rotation.getShuffled
				for (int i = 4; i > 1; i--) {
					int k = r.nextInt(i);
					int t = rots[i - 1];
					rots[i - 1] = rots[k];
					rots[k] = t;
				}
				for (int ri = 0; ri < 4 && !placed; ri++) {
					int tr = rots[ri];
					if (!mask[tr]) {
						skipShuffle(r, base0 < 0 ? 0 : cnt);
						continue;
					}
					int ntj = shuffledJigsaws(J, te, r, tgtJ);
					for (int b = 0; b < ntj; b++) {
						int k = tgtJ[b] < 0 ? 0 : tgtJ[b] - base0;
						if (!((mask[tr] >> k) & 1)) continue;
						JPlaced tj = placeJig(J, tgtJ[b], 0, 0, 0, tr);
						JBox bb;
						int rpy;
						connectBox(w, pk, g, J, te, tr, tj, sj, tx, ty, tz, fy, srcLocalY, srcRigid && TE.rigid, src.y0, S.expansion ? TE.expand[tr] : 0, ox, oz,
						           baseH, bb, rpy);
						if (n >= maxOut) return -1;
						Piece& p = out[n++];
						p.elem = (int16_t)te;
						p.rot = (uint8_t)tr;
						p.depth = (uint8_t)(depth + 1);
						p.x0 = (int16_t)bb.x0; p.y0 = (int16_t)bb.y0; p.z0 = (int16_t)bb.z0;
						p.x1 = (int16_t)bb.x1; p.y1 = (int16_t)bb.y1; p.z1 = (int16_t)bb.z1;
						p.region = (int16_t)region;
						p.prio = depth + 1 <= S.maxDepth ? (int16_t)sj.placement : (int16_t)-32768;
						placed = true;
						if (stopCat) {
							int cat = pieceCategory(J, p);
							if (cat >= 0 && ((stopCat[cat >> 6] >> (cat & 63)) & 1) && ++stopHave >= stopNeed) {
								*partial = true;
								return n;
							}
						}
						break;
					}
				}
			}
		}
	}
	return n;
}

// village-flavoured alias (the query side asks for villages)
HDN int villageLayout(const World& w, const PackDev& pk, const JigsawTables& J, int structureIndex, int chunkX, int chunkZ, Piece* out, int maxOut,
                      const HeightGrid* g = nullptr, const uint64_t* stopCat = nullptr, int stopNeed = 0, bool* partial = nullptr) {
	return jigsawLayout(w, pk, J, structureIndex, chunkX, chunkZ, out, maxOut, g, stopCat, stopNeed, partial);
}

HD int pieceTemplate(const JigsawTables& J, const Piece& p) { return J.elem[p.elem].kind == JK_FEATURE ? -1 : J.elem[p.elem].tpl; }
HD int pieceCategory(const JigsawTables& J, const Piece& p) {
	int t = J.elem[p.elem].kind == JK_SINGLE ? J.elem[p.elem].tpl : -1;
	return t >= 0 ? J.tpl[t].cat : -1;
}
HDN int countCategory(const JigsawTables& J, const Piece* p, int n, int cat) {
	int c = 0;
	for (int i = 0; i < n; i++) c += pieceCategory(J, p[i]) == cat;
	return c;
}
// world coordinates of a piece's box (minX, minY, minZ, maxX, maxY, maxZ)
HD void pieceBoxAbs(const Piece& p, int chunkX, int chunkZ, int* b) {
	b[0] = p.x0 + (chunkX << 4); b[1] = p.y0; b[2] = p.z0 + (chunkZ << 4);
	b[3] = p.x1 + (chunkX << 4); b[4] = p.y1; b[5] = p.z1 + (chunkZ << 4);
}

// ---------------------------------------------------------------- host: tables + building categories

// Building category of a template id, "" if it isn't a building. Villages only: templates under a
// .../houses/ or .../town_centers/ pool directory (zombie variants included). Strips the biome prefix and a
// trailing _N: village/plains/houses/plains_weaponsmith_1 -> weaponsmith, .../plains_small_house_3 -> small_house.
inline std::string templateCategory(const std::string& id) {
	std::string path = id.substr(id.find(':') + 1);
	std::vector<std::string> parts;
	for (size_t a = 0, b; a <= path.size(); a = b + 1) {
		b = path.find('/', a);
		if (b == std::string::npos) b = path.size();
		parts.push_back(path.substr(a, b - a));
	}
	if (parts.size() < 3 || parts[0] != "village") return "";
	bool building = false;
	for (size_t i = 1; i + 1 < parts.size(); i++) building |= parts[i] == "houses" || parts[i] == "town_centers";
	if (!building) return "";
	std::string n = parts.back(), biome = parts[1] + "_";
	if (n.rfind(biome, 0) == 0) n = n.substr(biome.size());
	size_t u = n.find_last_of('_');
	if (u != std::string::npos && u + 1 < n.size() && n.find_first_not_of("0123456789", u + 1) == std::string::npos) n = n.substr(0, u);
	return n;
}

struct JigsawHost {
	std::unique_ptr<JigsawTables> t;
	std::vector<std::string> categories;  // category index -> name
	std::vector<std::vector<int>> structCategories;  // jigsaw structure -> categories it can contain (sorted by name)
	std::vector<std::vector<int>> structTemplates;  // jigsaw structure -> reachable templates
	std::string error;  // why tables are unavailable (old pack, limits)
	int categoryIndex(const std::string& name) const {
		for (size_t i = 0; i < categories.size(); i++)
			if (categories[i] == name) return (int)i;
		return -1;
	}
};

inline JigsawHost buildJigsaw(const Pack& P) {
	JigsawHost H;
	const JigsawPackInfo& I = P.jigsaw;
	if (!I.present) {
		H.error = "pack has no jigsaw section (rebuild it with tools/extract.py)";
		return H;
	}
	H.t.reset(new JigsawTables());
	JigsawTables& J = *H.t;
	memset(&J, 0, sizeof J);
	auto need = [&](size_t n, int cap, const char* what) {
		if ((int)n > cap) throw std::runtime_error(std::string("jigsaw tables: too many ") + what);
	};
	need(I.templates.size(), JMAX_TEMPLATES, "templates");
	need(I.elements.size(), JMAX_ELEMENTS, "elements");
	need(I.pools.size(), JMAX_POOLS, "pools");
	need(I.structures.size(), JMAX_STRUCTS, "structures");
	J.featureName = -1;
	if (I.featureName != "*") {
		J.featureName = (int)I.names.size();  // not used by any template: matches no target unless interned
		for (size_t i = 0; i < I.names.size(); i++)
			if (I.names[i] == I.featureName) J.featureName = (int)i;
	}
	J.ntpl = (int)I.templates.size();
	for (int i = 0; i < J.ntpl; i++) {
		const auto& T = I.templates[i];
		JTpl& d = J.tpl[i];
		d.sx = (int16_t)T.sx;
		d.sy = (int16_t)T.sy;
		d.sz = (int16_t)T.sz;
		d.j0 = (uint16_t)J.njig;
		d.nj = (uint16_t)T.jig.size();
		need(T.jig.size(), JMAX_PJIG, "jigsaws in one template");
		need(J.njig + T.jig.size(), JMAX_JIGSAWS, "jigsaws");
		for (const auto& g : T.jig) {
			JJig& q = J.jig[J.njig++];
			q.x = (int16_t)g.x;
			q.y = (int16_t)g.y;
			q.z = (int16_t)g.z;
			q.front = (uint8_t)g.front;
			q.top = (uint8_t)g.top;
			q.rollable = g.joint == 0;
			q.name = (int16_t)g.name;
			q.target = (int16_t)g.target;
			q.pool = (int16_t)g.pool;
			q.placement = (int16_t)g.placement;
			q.selection = (int16_t)g.selection;
		}
		std::string c = templateCategory(T.name);
		d.cat = -1;
		if (!c.empty()) {
			int k = H.categoryIndex(c);
			if (k < 0) {
				k = (int)H.categories.size();
				H.categories.push_back(c);
			}
			d.cat = (int16_t)k;
		}
	}
	need(H.categories.size(), JMAX_CATS, "categories");
	J.ncat = (int)H.categories.size();
	J.nelem = (int)I.elements.size();
	for (int i = 0; i < J.nelem; i++) {
		const auto& E = I.elements[i];
		JElem& d = J.elem[i];
		d.kind = E.kind == "single" || E.kind == "legacy" ? JK_SINGLE : E.kind == "feature" ? JK_FEATURE : E.kind == "list" ? JK_LIST : JK_EMPTY;
		d.rigid = E.proj == "rigid";
		d.tpl = d.kind == JK_SINGLE ? (int16_t)std::stoi(E.arg) : (int16_t)-1;
		d.sub0 = (uint16_t)J.nsub;
		d.nsub = (uint16_t)E.subs.size();
		need(J.nsub + E.subs.size(), JMAX_SUBS, "list members");
		for (int s : E.subs) J.sub[J.nsub++] = (uint16_t)s;
	}
	for (int i = 0; i < J.nelem; i++)  // list template = first member's (for naming)
		if (J.elem[i].kind == JK_LIST && J.elem[i].nsub) J.elem[i].tpl = J.elem[J.sub[J.elem[i].sub0]].tpl;
	J.npool = (int)I.pools.size();
	int emptyPool = -1;
	for (int i = 0; i < J.npool; i++) {
		const auto& Q = I.pools[i];
		JPool& d = J.pool[i];
		if (Q.name == "minecraft:empty") emptyPool = i;
		d.e0 = (uint16_t)J.nent;
		for (auto& e : Q.entries) {
			need(J.nent + e.second, JMAX_ENTRIES, "pool entries");
			for (int k = 0; k < e.second; k++) J.ent[J.nent++] = (uint16_t)e.first;
		}
		d.n = (uint16_t)(J.nent - d.e0);
		d.fallback = (int16_t)Q.fallback;
		d.maxSize = (int16_t)Q.maxSize;
	}
	for (int i = 0; i < J.npool; i++) {
		JPool& d = J.pool[i];
		bool selfOk = d.n > 0 || i == emptyPool;
		bool fbOk = d.fallback >= 0 && (J.pool[d.fallback].n > 0 || d.fallback == emptyPool);
		d.ok = selfOk && fbOk;
		if (d.fallback < 0) d.fallback = (int16_t)i;  // never used (ok = 0)
	}
	// expansion hack heights per element and rotation (JigsawPlacement: targets' inward-facing jigsaws)
	for (int e = 0; e < J.nelem; e++) {
		JElem& E = J.elem[e];
		for (int rot = 0; rot < 4; rot++) {
			E.expand[rot] = 0;
			if (E.kind == JK_EMPTY) continue;
			JBox hb = elemBox(J, e, 0, 0, 0, rot);
			if (hb.y1 - hb.y0 + 1 > 16) continue;
			JavaRandom dummy;
			dummy.setSeed(0);
			int lst[JMAX_PJIG];
			int n = shuffledJigsaws(J, e, dummy, lst), m = 0;
			for (int k = 0; k < n; k++) {
				JPlaced p = placeJig(J, lst[k], 0, 0, 0, rot);
				int fx, fy, fz;
				dirVec(p.front, fx, fy, fz);
				int x = p.x + fx, y = p.y + fy, z = p.z + fz;
				if (x < hb.x0 || x > hb.x1 || y < hb.y0 || y > hb.y1 || z < hb.z0 || z > hb.z1) continue;
				int a = p.pool >= 0 ? J.pool[p.pool].maxSize : 0;
				int b = p.pool >= 0 && I.pools[p.pool].fallback >= 0 ? J.pool[I.pools[p.pool].fallback].maxSize : 0;
				int v = a > b ? a : b;
				m = v > m ? v : m;
			}
			E.expand[rot] = (int16_t)m;
		}
	}
	for (int i = 0; i < MAX_STRUCTS; i++) J.structOf[i] = -1;
	J.nstruct = (int)I.structures.size();
	H.structCategories.resize(J.nstruct);
	H.structTemplates.resize(J.nstruct);
	for (int i = 0; i < J.nstruct; i++) {
		const auto& S = I.structures[i];
		JStruct& d = J.st[i];
		d.startPool = (int16_t)S.startPool;
		d.startJigsaw = (int16_t)S.startJigsaw;
		d.maxDepth = (int16_t)S.size;
		d.maxH = (int16_t)S.maxH;
		d.maxV = (int16_t)std::min(S.maxV, 4064);
		d.padBottom = (int16_t)S.padBottom;
		d.padTop = (int16_t)S.padTop;
		d.heightKind = S.heightKind == "const" ? JH_CONST : S.heightKind == "uniform" ? JH_UNIFORM : JH_UNSUPPORTED;
		d.h0 = S.h0;
		d.h1 = S.h1;
		d.project = S.project != "-";
		d.expansion = (uint8_t)S.expansion;
		d.supported = !S.aliases && d.heightKind != JH_UNSUPPORTED && S.startPool >= 0;
		auto it = P.structureIndex.find(S.name);
		if (it != P.structureIndex.end() && it->second < MAX_STRUCTS) J.structOf[it->second] = i;
		// reachable pools/templates (start pool, jigsaw target pools, fallbacks)
		std::vector<char> seenP(J.npool, 0), seenT(J.ntpl, 0);
		std::vector<int> q;
		if (S.startPool >= 0) {
			q.push_back(S.startPool);
			seenP[S.startPool] = 1;
		}
		std::function<void(int)> addElem = [&](int e) {
			const JElem& E = J.elem[e];
			if (E.kind == JK_LIST)
				for (int k = 0; k < E.nsub; k++) addElem(J.sub[E.sub0 + k]);
			if (E.kind != JK_SINGLE || seenT[E.tpl]) return;
			seenT[E.tpl] = 1;
			const JTpl& T = J.tpl[E.tpl];
			for (int k = 0; k < T.nj; k++) {
				int p = J.jig[T.j0 + k].pool;
				if (p >= 0 && (int)J.pool[p].n + J.pool[J.pool[p].fallback].n > JMAX_SHUFFLE) d.supported = 0;  // list[] in jigsawLayout
				if (p >= 0 && !seenP[p]) {
					seenP[p] = 1;
					q.push_back(p);
				}
			}
		};
		for (size_t k = 0; k < q.size(); k++) {
			const JPool& Q = J.pool[q[k]];
			for (int x = 0; x < Q.n; x++) addElem(J.ent[Q.e0 + x]);
			int f = I.pools[q[k]].fallback;
			if (f >= 0 && !seenP[f]) {
				seenP[f] = 1;
				q.push_back(f);
			}
		}
		for (int t = 0; t < J.ntpl; t++)
			if (seenT[t]) {
				H.structTemplates[i].push_back(t);
				int c = J.tpl[t].cat;
				if (c >= 0 && std::find(H.structCategories[i].begin(), H.structCategories[i].end(), c) == H.structCategories[i].end())
					H.structCategories[i].push_back(c);
			}
		std::sort(H.structCategories[i].begin(), H.structCategories[i].end(), [&](int a, int b) { return H.categories[a] < H.categories[b]; });
	}
	return H;
}

// category names a pack structure can contain (e.g. for "minecraft:village_plains"), for list commands
inline std::vector<std::string> jigsawCategories(const JigsawHost& H, int structureIndex) {
	std::vector<std::string> out;
	if (!H.t || structureIndex < 0 || structureIndex >= MAX_STRUCTS) return out;
	int js = H.t->structOf[structureIndex];
	if (js < 0) return out;
	for (int c : H.structCategories[js]) out.push_back(H.categories[c]);
	return out;
}

// printable element id: template id, feature:<placed feature>, list:<first member>, empty
inline std::string elementName(const Pack& P, int e) {
	const auto& E = P.jigsaw.elements[e];
	if (E.kind == "feature") return "feature:" + E.arg;
	if (E.kind == "empty") return "empty";
	if (E.kind == "list") return "list:" + (E.subs.empty() ? std::string("?") : elementName(P, E.subs[0]));
	return P.jigsaw.templates[std::stoi(E.arg)].name;
}

}  // namespace sw

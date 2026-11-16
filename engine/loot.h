// Chest loot: which items a structure's loot containers hold, from the seed alone.
//
// A container placed by worldgen gets its LootTableSeed from the decoration random at the moment it is placed
// (StructureTemplate.placeInWorld / StructurePiece.createChest: random.nextLong()). That random is
// ChunkGenerator.applyBiomeDecoration's WorldgenRandom (Xoroshiro), set per chunk and per structure with
// setDecorationSeed(worldSeed, chunk min x, min z) + setFeatureSeed(.., index of the structure within its
// generation step in registry order, step). Within one chunk the pieces of the structure that touch it are
// placed in piece order, each template's containers in palette order, only those inside the chunk. The loot
// itself is LootTable.fill with LegacyRandomSource(seed): per pool a roll count, per roll a weighted entry,
// then its set_count; the final shuffle into slots doesn't change the totals.
//
// Exact as long as nothing else draws from that random before a container in the same chunk: feature pool
// elements (village decor: trees, hay piles, flowers) draw an amount that depends on the blocks around them,
// so containers after one in the same chunk are "unknown" (never counted). Tables with modifiers that use the
// random other than set_count (enchantments, ...) are unsupported: their containers are unknown too.
#pragma once
#include "jigsaw.h"
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace sw {

enum { LMAX_TABLES = 96, LMAX_POOLS = 512, LMAX_ENTRIES = 4096, LMAX_CONT = 4096, LMAX_ITEMS = 1024 };

struct LEntry {
	int16_t item;        // -1 = empty
	int16_t weight;      // with luck 0: max(floor(weight + quality * 0), 0)
	int16_t cmin, cmax;  // set_count range (1,1 without set_count)
	uint8_t add;         // set_count add=true: one item plus the count
};
struct LPool {
	int16_t rmin, rmax;
	uint16_t e0, ne;
};
struct LTable {
	uint16_t p0, np;
	uint8_t ok;  // 0: uses the random beyond what is modelled here (containers with it are unknown)
};
struct LCont {
	int16_t x, y, z;  // template-local position
	int16_t table;    // -1: no loot table (still draws its seed)
};

struct LootTables {
	int ntables, nitems, buried;  // buried: the buried treasure chest's table (-1 = none)
	int buriedSt;                 // pack structure index of minecraft:buried_treasure (-1 = none)
	LTable t[LMAX_TABLES];
	LPool p[LMAX_POOLS];
	LEntry e[LMAX_ENTRIES];
	uint16_t tc0[JMAX_TEMPLATES], tnc[JMAX_TEMPLATES];  // jigsaw template -> its containers in c[]
	LCont c[LMAX_CONT];
	int16_t step[MAX_STRUCTS], index[MAX_STRUCTS];  // pack structure -> setFeatureSeed (step, index); -1 unknown
};

// WorldgenRandom wrapping a XoroshiroRandomSource: next(bits) = top bits of the Xoroshiro long, and the
// legacy nextLong() built from two next(32).
struct WgRandom {
	Xoro x;
	HD void setSeed(uint64_t s) { x.seed64(s); }
	HD int next(int bits) { return (int)(int64_t)(x.nextLong() >> (64 - bits)); }
	HD int64_t nextLong() {
		int64_t hi = next(32);
		int64_t lo = next(32);
		return (int64_t)((uint64_t)hi << 32) + lo;
	}
};

// setDecorationSeed(seed, blockX, blockZ) then setFeatureSeed(result, index, step)
HD void lootRandom(WgRandom& r, int64_t worldSeed, int chunkX, int chunkZ, int step, int index) {
	r.setSeed((uint64_t)worldSeed);
	int64_t xs = r.nextLong() | 1, zs = r.nextLong() | 1;
	int64_t deco = (int64_t)((uint64_t)(int64_t)(chunkX * 16) * (uint64_t)xs + (uint64_t)(int64_t)(chunkZ * 16) * (uint64_t)zs) ^ worldSeed;
	r.setSeed((uint64_t)deco + (uint64_t)(int64_t)index + (uint64_t)(int64_t)(10000 * step));
}

// Mth.nextInt
HD int mthNextInt(JavaRandom& r, int lo, int hi) { return lo >= hi ? lo : r.nextInt(hi - lo + 1) + lo; }

// how many of `item` LootTable.fill(seed) puts in a container (luck 0); item -1: all items
HDN int lootCount(const LootTables& L, int table, int64_t seed, int item) {
	JavaRandom r;
	r.setSeed((uint64_t)seed);
	const LTable& T = L.t[table];
	int total = 0;
	for (int pi = 0; pi < T.np; pi++) {
		const LPool& P = L.p[T.p0 + pi];
		int rolls = mthNextInt(r, P.rmin, P.rmax);
		int wsum = 0, nvalid = 0, only = -1;
		for (int k = 0; k < P.ne; k++)
			if (L.e[P.e0 + k].weight > 0) wsum += L.e[P.e0 + k].weight, nvalid++, only = k;
		if (wsum == 0 || nvalid == 0) continue;
		for (int roll = 0; roll < rolls; roll++) {
			int pick = only;
			if (nvalid > 1) {
				int idx = r.nextInt(wsum);
				for (int k = 0; k < P.ne; k++) {
					int w = L.e[P.e0 + k].weight;
					if (w <= 0) continue;
					idx -= w;
					if (idx < 0) {
						pick = k;
						break;
					}
				}
			}
			const LEntry& E = L.e[P.e0 + pick];
			if (E.item < 0) continue;
			int n = (E.add ? 1 : 0) + mthNextInt(r, E.cmin, E.cmax);
			if (n > 0 && (item < 0 || E.item == item)) total += n;
		}
	}
	return total;
}

// can this table ever give the item? (unknown containers only matter when it can)
HDN bool lootCanGive(const LootTables& L, int table, int item) {
	if (table < 0) return false;
	const LTable& T = L.t[table];
	if (!T.ok) return true;
	for (int pi = 0; pi < T.np; pi++) {
		const LPool& P = L.p[T.p0 + pi];
		if (P.rmax <= 0) continue;
		for (int k = 0; k < P.ne; k++) {
			const LEntry& E = L.e[P.e0 + k];
			if (E.item == item && E.weight > 0 && (E.add || E.cmax > 0)) return true;
		}
	}
	return false;
}

// Buried treasure: one chest at (chunk*16 + 9, chunk*16 + 9), the first draw of its chunk (BuriedTreasurePieces).
HDN int buriedLoot(const LootTables& L, int st, int64_t worldSeed, int chunkX, int chunkZ, int item) {
	if (L.buried < 0 || L.step[st] < 0 || !L.t[L.buried].ok) return -1;
	WgRandom r;
	lootRandom(r, worldSeed, chunkX, chunkZ, L.step[st], L.index[st]);
	return lootCount(L, L.buried, r.nextLong(), item);
}

// One loot container of a jigsaw structure as placed: world position, table, seed (valid when known).
struct LootChest {
	int x, y, z, table;
	int64_t seed;
	uint8_t known;
};

// Every loot container of a laid-out jigsaw structure (pieces relative to the start chunk, as jigsawLayout gives
// them), in placement order per chunk: f(const LootChest&). Returns false if the chunk list overflowed.
// placement units of a piece: a single element, or a list's members in order (ListPoolElement.place)
HD int pieceUnits(const JigsawTables& J, const Piece& p, int* u) {
	const JElem& E = J.elem[p.elem];
	if (E.kind != JK_LIST) {
		u[0] = p.elem;
		return 1;
	}
	int n = 0;
	for (int k = 0; k < E.nsub && n < 16; k++) u[n++] = J.sub[E.sub0 + k];
	return n;
}

template <class F>
HDN bool forEachChest(const LootTables& L, const JigsawTables& J, int st, const Piece* pcs, int npcs, int startCx, int startCz, int64_t worldSeed, F f) {
	if (L.step[st] < 0) return true;
	enum { MAXCH = 128 };
	int chx[MAXCH], chz[MAXCH], nch = 0;
	bool ok = true;
	for (int i = 0; i < npcs; i++) {  // chunks that hold a container
		int u[16], nu = pieceUnits(J, pcs[i], u), px, py, pz;
		pieceOrigin(J, pcs[i], px, py, pz);
		for (int k = 0; k < nu; k++) {
			if (J.elem[u[k]].kind != JK_SINGLE) continue;
			int t = J.elem[u[k]].tpl;
			for (int c = 0; c < L.tnc[t]; c++) {
				const LCont& C = L.c[L.tc0[t] + c];
				int rx, rz;
				rotXZ(pcs[i].rot, C.x, C.z, rx, rz);
				int x = ((startCx << 4) + px + rx) >> 4, z = ((startCz << 4) + pz + rz) >> 4;
				bool seen = false;
				for (int e = 0; e < nch && !seen; e++) seen = chx[e] == x && chz[e] == z;
				if (seen) continue;
				if (nch == MAXCH) ok = false;
				else chx[nch] = x, chz[nch++] = z;
			}
		}
	}
	for (int ci = 0; ci < nch; ci++) {
		WgRandom r;
		lootRandom(r, worldSeed, chx[ci], chz[ci], L.step[st], L.index[st]);
		bool dirty = false;
		int bx0 = chx[ci] << 4, bz0 = chz[ci] << 4;
		for (int i = 0; i < npcs; i++) {
			int b[6];
			pieceBoxAbs(pcs[i], startCx, startCz, b);
			if (b[3] < bx0 || b[0] > bx0 + 15 || b[5] < bz0 || b[2] > bz0 + 15) continue;  // StructureStart.placeInChunk
			int u[16], nu = pieceUnits(J, pcs[i], u), px, py, pz;
			pieceOrigin(J, pcs[i], px, py, pz);
			for (int k = 0; k < nu; k++) {
				const JElem& E = J.elem[u[k]];
				if (E.kind == JK_FEATURE) dirty = true;  // a feature draws an unknown amount from here on
				if (E.kind != JK_SINGLE) continue;
				for (int c = 0; c < L.tnc[E.tpl]; c++) {
					const LCont& C = L.c[L.tc0[E.tpl] + c];
					int rx, rz;
					rotXZ(pcs[i].rot, C.x, C.z, rx, rz);
					int wx = (startCx << 4) + px + rx, wz = (startCz << 4) + pz + rz;
					if ((wx >> 4) != chx[ci] || (wz >> 4) != chz[ci]) continue;
					LootChest o;
					o.x = wx, o.y = py + C.y, o.z = wz, o.table = C.table, o.known = !dirty, o.seed = dirty ? 0 : r.nextLong();
					f(o);
				}
			}
		}
	}
	return ok;
}

struct ChestCollector {
	LootChest* out;
	int max, *n;
	HD void operator()(const LootChest& c) const {
		if (*n < max) out[(*n)++] = c;
	}
};
inline int jigsawChests(const LootTables& L, const JigsawTables& J, int st, const Piece* pcs, int npcs, int startCx, int startCz, int64_t worldSeed,
                        LootChest* out, int maxOut) {
	int n = 0;
	forEachChest(L, J, st, pcs, npcs, startCx, startCz, worldSeed, ChestCollector{out, maxOut, &n});
	return n;
}

// ---------------------------------------------------------------- host: pack section

struct LootHost {
	bool ok = false;
	bool exactLayouts = false;  // the pack has exact terrain (26.3+): village layouts, so village chests, are exact
	std::vector<std::string> items, tables;
	std::unique_ptr<LootTables> t;
	int item(const std::string& name) const {
		for (size_t i = 0; i < items.size(); i++)
			if (items[i] == name || items[i] == "minecraft:" + name) return (int)i;
		return -1;
	}
};

// Reads the "loot" section (tools/extract.py) of a pack; structNames: pack structure names in pack order,
// tplNames: jigsaw template names in jigsaw table order.
inline void loadLoot(const std::string& path, const std::vector<std::string>& structNames, const std::vector<std::string>& tplNames, LootHost& H) {
	std::ifstream in(path);
	std::string line;
	while (std::getline(in, line))
		if (line.rfind("loot 1", 0) == 0) break;
	if (!in) return;
	{  // exact terrain section present? (village layouts are only estimates without it: their chests would be wrong)
		std::ifstream t(path);
		std::string l;
		while (std::getline(t, l))
			if (l.rfind("terrain ", 0) == 0) H.exactLayouts = true;
	}
	auto L = std::make_unique<LootTables>();
	memset(L.get(), 0, sizeof(LootTables));
	for (auto& s : L->step) s = -1;
	for (auto& s : L->index) s = -1;
	L->buriedSt = -1;
	{
		std::istringstream ss(line);
		std::string k;
		int ver;
		ss >> k >> ver >> L->buried;
	}
	auto header = [&](const char* name) {
		std::getline(in, line);
		std::istringstream ss(line);
		std::string k;
		int n = 0;
		ss >> k >> n;
		if (k != name) throw std::runtime_error(std::string("loot section: expected ") + name);
		return n;
	};
	int n = header("loot_structures");
	for (int i = 0; i < n; i++) {
		std::getline(in, line);
		std::istringstream ss(line);
		std::string name;
		int step, idx;
		ss >> name >> step >> idx;
		for (size_t s = 0; s < structNames.size() && s < MAX_STRUCTS; s++)
			if (structNames[s] == name) {
				L->step[s] = (int16_t)step, L->index[s] = (int16_t)idx;
				if (name == "minecraft:buried_treasure") L->buriedSt = (int)s;
			}
	}
	n = header("loot_templates");
	int nc = 0;
	for (int i = 0; i < n; i++) {
		std::getline(in, line);
		std::istringstream ss(line);
		int t, k;
		ss >> t >> k;
		if (t < 0 || t >= JMAX_TEMPLATES || nc + k > LMAX_CONT) throw std::runtime_error("loot section: too many containers");
		L->tc0[t] = (uint16_t)nc;
		L->tnc[t] = (uint16_t)k;
		for (int c = 0; c < k; c++) {
			LCont& C = L->c[nc++];
			ss >> C.x >> C.y >> C.z >> C.table;
		}
	}
	(void)tplNames;
	n = header("loot_tables");
	if (n > LMAX_TABLES) throw std::runtime_error("loot section: too many tables");
	L->ntables = n;
	int np = 0, ne = 0;
	for (int i = 0; i < n; i++) {
		std::getline(in, line);
		std::istringstream ss(line);
		std::string name;
		int ok, pools;
		ss >> name >> ok >> pools;
		H.tables.push_back(name);
		LTable& T = L->t[i];
		T.ok = (uint8_t)ok;
		T.p0 = (uint16_t)np;
		T.np = (uint16_t)pools;
		for (int p = 0; p < pools; p++) {
			if (np >= LMAX_POOLS) throw std::runtime_error("loot section: too many pools");
			LPool& P = L->p[np++];
			int k;
			ss >> P.rmin >> P.rmax >> k;
			P.e0 = (uint16_t)ne;
			P.ne = (uint16_t)k;
			for (int e = 0; e < k; e++) {
				if (ne >= LMAX_ENTRIES) throw std::runtime_error("loot section: too many entries");
				LEntry& E = L->e[ne++];
				int item, w, q, a, b, add;
				ss >> item >> w >> q >> a >> b >> add;
				E.item = (int16_t)item;
				E.weight = (int16_t)(w > 0 ? w : 0);  // luck 0: quality doesn't matter
				E.cmin = (int16_t)a, E.cmax = (int16_t)b, E.add = (uint8_t)add;
				(void)q;
			}
		}
	}
	n = header("loot_items");
	for (int i = 0; i < n; i++) {
		std::getline(in, line);
		H.items.push_back(line);
	}
	L->nitems = n;
	H.t = std::move(L);
	H.ok = true;
}

}  // namespace sw

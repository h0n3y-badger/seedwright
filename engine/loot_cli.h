// Chest loot commands:
//   seedwright <pack> loot-verify <oracle loot-<seed>.txt ...>   compare with the oracle's loot dump (26.3+)
//   seedwright <pack> loot <seed> <structure> <chunkX> <chunkZ>    the loot containers of one structure start
#pragma once
#include "loot.h"
#include "terrain.h"
#include <map>

namespace sw {

inline LootHost buildLoot(const Pack& P, const char* path) {
	LootHost H;
	std::vector<std::string> names;
	for (auto& s : P.structures) names.push_back(s.name);
	std::vector<std::string> tpl;
	for (auto& t : P.jigsaw.templates) tpl.push_back(t.name);
	loadLoot(path, names, tpl, H);
	return H;
}

// item totals of a container: (item name -> count), via lootCount per item
inline std::map<std::string, int> lootTotals(const LootHost& H, int table, int64_t seed) {
	std::map<std::string, int> m;
	for (int i = 0; i < (int)H.items.size(); i++) {
		int n = lootCount(*H.t, table, seed, i);
		if (n) m[H.items[i]] = n;
	}
	return m;
}

inline std::string totalsText(const std::map<std::string, int>& m) {
	std::string s;
	for (auto& kv : m) s += " " + kv.first + "=" + std::to_string(kv.second);
	return s;
}

// exact layout + chests of one jigsaw start (TR: exact terrain; null = estimated heights)
inline int startChests(const Pack& P, const JigsawTables& J, const LootHost& H, const Terrain* TR, int64_t seed, int st, int cx, int cz,
                       std::vector<LootChest>& out) {
	World* w = new World();
	worldInit(*w, *P.dev, seed);
	std::vector<Piece> pcs(1024);
	int n;
	if (TR) {
		TerrainWorld tw(*TR, seed);
		HeightGrid g = terrainGrid(tw);
		n = villageLayout(*w, *P.dev, J, st, cx, cz, pcs.data(), (int)pcs.size(), &g);
	} else {
		n = villageLayout(*w, *P.dev, J, st, cx, cz, pcs.data(), (int)pcs.size(), nullptr);
	}
	delete w;
	out.resize(512);
	int k = n > 0 ? jigsawChests(*H.t, J, st, pcs.data(), n, cx, cz, seed, out.data(), (int)out.size()) : 0;
	out.resize(k);
	return n;
}

inline int lootCommand(const Pack& P, const JigsawTables* J, int argc, char** argv) {
	std::string cmd = argv[2];
	LootHost H = buildLoot(P, argv[1]);
	if (!H.ok) {
		fprintf(stderr, "this pack has no loot section (rebuild it with tools/build-packs.sh)\n");
		return 2;
	}
	if (!J) {
		fprintf(stderr, "this pack has no jigsaw tables\n");
		return 2;
	}
	Terrain TR;
	bool exactTerrain = TR.load(argv[1]);
	if (cmd == "loot") {
		if (argc < 7) {
			fprintf(stderr, "usage: seedwright <pack> loot <seed> <structure> <chunkX> <chunkZ>\n");
			return 2;
		}
		int64_t seed = std::stoll(argv[3]);
		std::string name = argv[4];
		if (name.find(':') == std::string::npos) name = "minecraft:" + name;
		auto it = P.structureIndex.find(name);
		if (it == P.structureIndex.end()) {
			fprintf(stderr, "unknown structure %s\n", argv[4]);
			return 2;
		}
		int st = it->second, cx = atoi(argv[5]), cz = atoi(argv[6]);
		if (H.t->buried >= 0 && name == "minecraft:buried_treasure") {
			WgRandom r;
			lootRandom(r, seed, cx, cz, H.t->step[st], H.t->index[st]);
			int64_t s = r.nextLong();
			printf("container %d ? %d %s %lld%s\n", cx * 16 + 9, cz * 16 + 9, H.tables[H.t->buried].c_str(), (long long)s,
			       totalsText(lootTotals(H, H.t->buried, s)).c_str());
			return 0;
		}
		std::vector<LootChest> ch;
		startChests(P, *J, H, exactTerrain ? &TR : nullptr, seed, st, cx, cz, ch);
		for (auto& c : ch) {
			printf("container %d %d %d %s ", c.x, c.y, c.z, c.table >= 0 ? H.tables[c.table].c_str() : "none");
			if (!c.known) printf("unknown (a feature piece draws before it in its chunk)\n");
			else if (c.table >= 0 && !H.t->t[c.table].ok) printf("%lld unsupported table\n", (long long)c.seed);
			else printf("%lld%s\n", (long long)c.seed, c.table >= 0 ? totalsText(lootTotals(H, c.table, c.seed)).c_str() : "");
		}
		if (!exactTerrain) fprintf(stderr, "warning: no exact terrain for this version: estimated layout, chests may be wrong\n");
		return 0;
	}
	// loot-verify
	long long nOurs = 0, matched = 0, known = 0, seedOk = 0, lootOk = 0, unknown = 0, extra = 0, missing = 0;
	long long bTot = 0, bOk = 0;
	bool verbose = getenv("SW_VERBOSE") != nullptr;
	for (int f = 3; f < argc; f++) {
		std::ifstream in(argv[f]);
		std::string line, k;
		int64_t seed = 0;
		struct TC { int x, y, z; std::string table; int64_t seed; std::map<std::string, int> items; };
		struct TS { std::string name; int cx, cz; std::vector<TC> c; };
		std::vector<TS> starts;
		while (std::getline(in, line)) {
			std::istringstream ss(line);
			ss >> k;
			if (k == "seed") ss >> seed;
			else if (k == "structure") {
				TS s;
				ss >> s.name >> s.cx >> s.cz;
				starts.push_back(s);
			} else if (k == "container" && !starts.empty()) {
				TC c;
				ss >> c.x >> c.y >> c.z >> c.table >> c.seed;
				std::string it;
				while (ss >> it) {
					size_t e = it.find('=');
					c.items[it.substr(0, e)] = std::stoi(it.substr(e + 1));
				}
				starts.back().c.push_back(c);
			}
		}
		printf("== %s (seed %lld): %zu starts\n", argv[f], (long long)seed, starts.size());
		for (auto& s : starts) {
			auto it = P.structureIndex.find(s.name);
			if (it == P.structureIndex.end()) continue;
			int st = it->second;
			if (s.name == "minecraft:buried_treasure") {
				for (auto& c : s.c) {
					if (c.table != "minecraft:chests/buried_treasure" || c.x != s.cx * 16 + 9 || c.z != s.cz * 16 + 9) continue;
					WgRandom r;
					lootRandom(r, seed, s.cx, s.cz, H.t->step[st], H.t->index[st]);
					int64_t sd = r.nextLong();
					auto ours = lootTotals(H, H.t->buried, sd);
					bTot++;
					bool ok = sd == c.seed && ours == c.items;
					bOk += ok;
					if (!ok || verbose)
						printf("  buried %d %d: seed %lld vs %lld%s\n    ours:%s\n    game:%s\n", s.cx, s.cz, (long long)sd, (long long)c.seed, ok ? "" : "  MISMATCH",
						       totalsText(ours).c_str(), totalsText(c.items).c_str());
				}
				continue;
			}
			std::vector<LootChest> ch;
			int np = startChests(P, *J, H, exactTerrain ? &TR : nullptr, seed, st, s.cx, s.cz, ch);
			std::vector<char> used(s.c.size(), 0);
			int sMatched = 0, sBad = 0;
			for (auto& c : ch) {
				nOurs++;
				int m = -1;
				for (size_t i = 0; i < s.c.size() && m < 0; i++)
					if (!used[i] && s.c[i].x == c.x && s.c[i].z == c.z) m = (int)i;
				if (m < 0) {
					extra++;
					sBad++;
					printf("  %s %d %d: our container at %d %d %d (%s) not in the game\n", s.name.c_str(), s.cx, s.cz, c.x, c.y, c.z,
					       c.table >= 0 ? H.tables[c.table].c_str() : "none");
					continue;
				}
				used[m] = 1;
				matched++;
				sMatched++;
				const TC& t = s.c[m];
				if (!c.known) {
					unknown++;
					continue;
				}
				known++;
				bool sOk = c.seed == t.seed;
				seedOk += sOk;
				bool lOk = false;
				if (c.table >= 0 && H.t->t[c.table].ok) {
					auto ours = lootTotals(H, c.table, c.seed);
					lOk = sOk && ours == t.items;
					if (sOk && !lOk) printf("  %s %d %d: loot differs at %d %d %d\n    ours:%s\n    game:%s\n", s.name.c_str(), s.cx, s.cz, c.x, c.y, c.z,
					                       totalsText(ours).c_str(), totalsText(t.items).c_str());
				}
				lootOk += lOk;
				if (!sOk) {
					sBad++;
					printf("  %s %d %d: seed differs at %d %d %d: ours %lld game %lld\n", s.name.c_str(), s.cx, s.cz, c.x, c.y, c.z, (long long)c.seed,
					       (long long)t.seed);
				}
			}
			// game containers of village loot tables we don't have
			for (size_t i = 0; i < s.c.size(); i++) {
				if (used[i]) continue;
				bool ours = false;
				for (auto& tn : H.tables) ours |= tn == s.c[i].table;
				if (!ours || s.c[i].table.find("/village/") == std::string::npos) continue;  // dungeon, mineshaft... chests in the box
				missing++;
				sBad++;
				printf("  %s %d %d: game container at %d %d %d (%s) missing\n", s.name.c_str(), s.cx, s.cz, s.c[i].x, s.c[i].y, s.c[i].z, s.c[i].table.c_str());
			}
			printf("  %s %d %d: %d pieces, %zu containers, %d matched%s\n", s.name.c_str(), s.cx, s.cz, np, ch.size(), sMatched, sBad ? "  <-- differences" : "");
		}
	}
	printf("jigsaw containers: %lld ours, %lld matched by position, %lld extra, %lld missing; of the matched: %lld known (%lld seed ok, %lld loot ok), %lld unknown (feature before it)\n",
	       nOurs, matched, extra, missing, known, seedOk, lootOk, unknown);
	printf("buried treasure: %lld chests, %lld exact\n", bTot, bOk);
	bool ok = extra == 0 && missing == 0 && seedOk == known && bOk == bTot;
	printf("LOOT-VERIFY: %s\n", ok ? "all known containers exact" : "DIFFERENCES");
	return ok ? 0 : 1;
}

}  // namespace sw

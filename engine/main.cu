// seedwright CLI. Builds as CUDA (nvcc) or CPU-only (g++ -x c++ -DSW_CPU_ONLY).
#include "pack.h"
#include "structures.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <set>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace sw;

static StructTables buildTables(const Pack& P) {
	StructTables T;
	memset(&T, 0, sizeof T);
	T.nsets = (int)P.sets.size();
	T.nstructs = (int)P.structures.size();
	for (int i = 0; i < T.nstructs; i++) {
		for (int b : P.structures[i].biomes) T.st[i].biomes[b >> 6] |= 1ULL << (b & 63);
		T.st[i].jigsaw = P.structures[i].type == "jigsaw";
		const std::string& y = P.structures[i].y;
		T.st[i].check = P.structures[i].type == "mineshaft" ? CHECK_MINESHAFT : (y != "surface" ? CHECK_FIXED_Y : CHECK_CENTER);
		T.st[i].y = y == "surface" ? 0 : std::stoi(y);
	}
	for (int i = 0; i < T.nsets; i++) {
		const StructureSetInfo& s = P.sets[i];
		SetDev& d = T.sets[i];
		d.type = s.type == "random_spread" ? SET_SPREAD : s.type == "concentric_rings" ? SET_RINGS : SET_OTHER;
		d.spread = s.spread;
		d.exclusion = s.exclusionIndex;
		d.exclusionChunks = s.exclusionChunks;
		d.nstructs = (int)s.structs.size();
		for (int k = 0; k < d.nstructs; k++) {
			d.structs[k] = s.structs[k];
			d.weights[k] = s.weights[k];
		}
		d.distance = s.distance;
		d.ringSpread = s.ringSpread;
		d.count = s.count;
		for (size_t b = 0; b < s.preferredMask.size(); b++)
			if (s.preferredMask[b]) d.preferred[b >> 6] |= 1ULL << (b & 63);
	}
	return T;
}

static int setIndex(const Pack& P, const std::string& name) {
	std::string n = name.substr(name.find(':') + 1);
	for (size_t i = 0; i < P.sets.size(); i++)
		if (P.sets[i].name == n) return (int)i;
	return -1;
}

static uint32_t fbits(float f) {
	uint32_t u;
	memcpy(&u, &f, 4);
	return u;
}

// ------------------------------------------------------------------ verify against oracle dumps

static int verify(const Pack& P, const StructTables& T, const std::vector<std::string>& files) {
	const PackDev& pk = *P.dev;
	int bad = 0;
	for (const std::string& f : files) {
		std::ifstream in(f);
		std::string k, line;
		int64_t seed;
		in >> k >> seed;
		auto* w = new World();
		worldInit(*w, pk, seed);
		printf("== %s (seed %lld)\n", f.c_str(), (long long)seed);
		int n;
		in >> k >> n;  // points
		int climBad[6] = {0}, biomeBad = 0;
		for (int i = 0; i < n; i++) {
			int qx, qy, qz;
			std::string h[6], biome;
			in >> qx >> qy >> qz;
			for (auto& s : h) in >> s;
			in >> biome;
			float c[6];
			int b = biomeAt(*w, pk, qx, qy, qz, 0, c);
			for (int j = 0; j < 6; j++)
				if (fbits(c[j]) != (uint32_t)std::stoul(h[j], nullptr, 16)) {
					if (climBad[j]++ < 3)
						printf("  climate[%d] @%d,%d,%d: got %08x (%.9g) want %s (%.9g)\n", j, qx, qy, qz, fbits(c[j]), c[j], h[j].c_str(), bitsF(h[j]));
				}
			if (P.biomes[b] != biome && biomeBad++ < 5) printf("  biome @%d,%d,%d: got %s want %s\n", qx, qy, qz, P.biomes[b].c_str(), biome.c_str());
		}
		printf("  points: %d  climate mismatches t/h/c/e/d/w = %d %d %d %d %d %d  biome mismatches = %d\n", n, climBad[0], climBad[1], climBad[2],
		       climBad[3], climBad[4], climBad[5], biomeBad);
		bad += biomeBad;
		for (int j = 0; j < 6; j++) bad += climBad[j];
		in >> k >> n;  // columns
		int colBad = 0, colTot = 0;
		for (int i = 0; i < n; i++) {
			int qx, qz, q0, q1;
			in >> qx >> qz >> q0 >> q1;
			Column col;
			columnBiomes(*w, pk, qx, qz, col);
			for (int qy = q0; qy <= q1; qy++) {
				std::string b;
				in >> b;
				colTot++;
				if (P.biomes[col.biome[qy - col.qminY]] != b && colBad++ < 5)
					printf("  column biome @%d,%d,%d: got %s want %s\n", qx, qy, qz, P.biomes[col.biome[qy - col.qminY]].c_str(), b.c_str());
			}
		}
		printf("  column (volume) biomes: %d/%d mismatches\n", colBad, colTot);
		bad += colBad;
		int ox, oz;
		in >> k >> ox >> oz;
		int sx, sz;
		spawnOrigin(*w, pk, sx, sz);
		printf("  spawn origin chunk: got %d %d want %d %d %s\n", sx >> 4, sz >> 4, ox, oz, (sx >> 4) == ox && (sz >> 4) == oz ? "OK" : "MISMATCH");
		bad += !((sx >> 4) == ox && (sz >> 4) == oz);
		std::getline(in, line);
		std::getline(in, line);  // spawn (final, terrain-dependent)
		printf("  (vanilla final spawn: %s)\n", line.c_str());
		std::set<std::tuple<std::string, int, int>> wantC, gotC;
		int radius = 0;
		while (in >> k) {
			if (k == "rings") {
				std::string name;
				in >> name >> n;
				std::vector<int> wx(n), wz(n);
				for (int i = 0; i < n; i++) in >> wx[i] >> wz[i];
				int si = setIndex(P, name);
				int want = 12, gx[128], gz[128];
				strongholds(*w, pk, T.sets[si], want, gx, gz);
				int ok = 0;
				for (int i = 0; i < want; i++) ok += gx[i] == wx[i] && gz[i] == wz[i];
				printf("  strongholds (first %d of %d): %d/%d match%s\n", want, n, ok, want, ok == want ? "" : "  <-- MISMATCH");
				if (ok != want)
					for (int i = 0; i < want; i++) printf("    #%d got %d %d want %d %d\n", i, gx[i], gz[i], wx[i], wz[i]);
				bad += want - ok;
			} else if (k == "candidates") {
				in >> n;
				for (int i = 0; i < n; i++) {
					std::string s;
					int cx, cz;
					in >> s >> cx >> cz;
					wantC.insert({s.substr(s.find(':') + 1), cx, cz});
					radius = std::max(radius, std::max(std::abs(cx), std::abs(cz)));
				}
			} else if (k == "starts") {
				break;
			}
		}
		radius = 40;
		for (int si = 0; si < T.nsets; si++)
			for (int cx = -radius; cx <= radius; cx++)
				for (int cz = -radius; cz <= radius; cz++)
					if (isStructureChunk(T, si, (uint64_t)seed, cx, cz)) gotC.insert({P.sets[si].name, cx, cz});
		int missing = 0, extra = 0;
		for (auto& c : wantC)
			if (!gotC.count(c) && missing++ < 5) printf("  missing candidate %s %d %d\n", std::get<0>(c).c_str(), std::get<1>(c), std::get<2>(c));
		for (auto& c : gotC)
			if (!wantC.count(c) && extra++ < 5) printf("  extra candidate %s %d %d\n", std::get<0>(c).c_str(), std::get<1>(c), std::get<2>(c));
		printf("  placement candidates: %zu vanilla, %zu ours, %d missing, %d extra\n", wantC.size(), gotC.size(), missing, extra);
		bad += missing + extra;
		// starts: vanilla structure starts vs our biome-validated picks
		in >> n;
		std::set<std::tuple<std::string, int, int>> wantS, gotS;
		for (int i = 0; i < n; i++) {
			std::string s;
			int cx, cz, b[6];
			in >> s >> cx >> cz;
			for (int& v : b) in >> v;
			wantS.insert({s, cx, cz});
		}
		for (auto& c : gotC) {
			int si = setIndex(P, std::get<0>(c));
			if (T.sets[si].type != SET_SPREAD) continue;
			int st = structureAt(*w, pk, T, si, std::get<1>(c), std::get<2>(c));
			if (st >= 0) gotS.insert({P.structures[st].name, std::get<1>(c), std::get<2>(c)});
		}
		int sm = 0, se = 0;
		std::map<std::string, std::pair<int, int>> perType;
		for (auto& c : wantS) {
			if (std::get<0>(c) == "minecraft:stronghold") continue;
			perType[std::get<0>(c)].first++;
			if (!gotS.count(c)) {
				perType[std::get<0>(c)].second++;
				if (sm++ < 8) printf("  missed start %s %d %d\n", std::get<0>(c).c_str(), std::get<1>(c), std::get<2>(c));
			}
		}
		for (auto& c : gotS)
			if (!wantS.count(c)) {
				perType[std::get<0>(c)].second++;
				if (se++ < 8) printf("  false start %s %d %d\n", std::get<0>(c).c_str(), std::get<1>(c), std::get<2>(c));
			}
		printf("  structure starts (approx. surface height): %zu vanilla, %zu ours, %d missed, %d false\n", wantS.size(), gotS.size(), sm, se);
		delete w;
	}
	printf("%s\n", bad ? "VERIFY: MISMATCHES (exact parts)" : "VERIFY: all exact parts bit-identical");
	return bad ? 1 : 0;
}

int main(int argc, char** argv) {
	if (argc < 3) {
		fprintf(stderr, "usage: seedwright <pack> verify <oracle files...>\n       seedwright <pack> search ...\n");
		return 2;
	}
	Pack P = loadPack(argv[1]);
	StructTables T = buildTables(P);
	std::string cmd = argv[2];
	if (cmd == "verify") return verify(P, T, std::vector<std::string>(argv + 3, argv + argc));
	fprintf(stderr, "unknown command %s\n", cmd.c_str());
	return 2;
}

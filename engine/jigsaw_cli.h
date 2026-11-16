// Debug CLI for jigsaw layouts (dispatched from main.cu for every command starting with "village"):
//   seedwright <pack> village <seed> <blockX> <blockZ> [--heights villages-file]   pieces + building counts
//   seedwright <pack> village-categories                                           "piece <structure> <category>"
//   seedwright <pack> village-verify <villages-file...>                            compare with oracle dumps
//   seedwright <pack> village-bench [N]                                            single-thread cost per village
#pragma once
#include "jigsaw.h"
#include "terrain.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <sstream>

namespace sw {

namespace jcli {

inline double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

inline std::string shortName(const std::string& n) { return n.substr(n.find(':') + 1); }

// one start from an oracle villages-<seed>.txt
struct OracleStart {
	std::string structure;
	int cx = 0, cz = 0;
	std::vector<std::string> elem;
	std::vector<int> rot;
	std::vector<std::array<int, 6>> box;
	int hx0 = 0, hz0 = 0, hnx = 0, hnz = 0;
	std::vector<int16_t> heights;
	HeightGrid grid() const { return HeightGrid{heights.data(), hx0, hz0, hnx, hnz}; }
};

inline bool readOracle(const std::string& path, int64_t& seed, std::vector<OracleStart>& out) {
	std::ifstream in(path);
	if (!in) return false;
	std::string k;
	in >> k >> seed;
	while (in >> k) {
		if (k == "end") break;
		if (k != "structure") return false;
		OracleStart s;
		int n;
		in >> s.structure >> s.cx >> s.cz >> n;
		for (int i = 0; i < n; i++) {
			std::string e, r;
			std::array<int, 6> b;
			in >> e >> r;
			for (int& v : b) in >> v;
			s.elem.push_back(e);
			s.rot.push_back(r == "?" ? -1 : std::stoi(r));
			s.box.push_back(b);
		}
		in >> k >> s.hx0 >> s.hz0 >> s.hnx >> s.hnz;
		s.heights.resize((size_t)s.hnx * s.hnz);
		for (auto& h : s.heights) {
			std::string t;
			in >> t;
			h = t == "-" ? (int16_t)HEIGHT_UNKNOWN : (int16_t)std::stoi(t);
		}
		out.push_back(std::move(s));
	}
	return true;
}

inline int setOf(const Pack& P, int st) {
	for (size_t i = 0; i < P.sets.size(); i++)
		for (int s : P.sets[i].structs)
			if (s == st) return (int)i;
	return -1;
}

inline std::string pieceDesc(const Pack& P, const Piece& p, int ox, int oz) {
	char b[160];
	snprintf(b, sizeof b, "%d %d %d %d %d %d", p.x0 + ox, p.y0, p.z0 + oz, p.x1 + ox, p.y1, p.z1 + oz);
	return elementName(P, p.elem) + " " + std::to_string(p.rot) + " " + b;
}

inline std::map<std::string, int> categoryCounts(const JigsawHost& H, const Piece* p, int n) {
	std::map<std::string, int> m;
	for (int i = 0; i < n; i++) {
		int c = pieceCategory(*H.t, p[i]);
		if (c >= 0) m[H.categories[c]]++;
	}
	return m;
}
inline std::map<std::string, int> categoryCounts(const OracleStart& s) {
	std::map<std::string, int> m;
	for (auto& e : s.elem) {
		std::string c = templateCategory(e);
		if (!c.empty()) m[c]++;
	}
	return m;
}

enum { CLI_MAX_PIECES = 1024, GPU_MAX_PIECES = 400 };

// bench seeds: candidate chunk of village region (0,0) for seed i
HD int64_t benchSeed(int i) { return (int64_t)mixStafford13((uint64_t)i + 777); }

#ifndef SW_CPU_ONLY
#define JCK(x)                                                                                                  \
	do {                                                                                                        \
		cudaError_t e_ = (x);                                                                                   \
		if (e_ != cudaSuccess) {                                                                                \
			fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);                  \
			exit(1);                                                                                            \
		}                                                                                                       \
	} while (0)

// one thread per seed: village start + layout at the seed's region (0,0) candidate; out = piece count
// (-2 = no village there), cat = pieces of category `cat`
__global__ void kJigsawLayout(const PackDev* pk, const StructTables* T, const JigsawTables* J, int set, int n, World* worlds, Piece* pieces, int* out,
                              int* catOut, int cat) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	int64_t seed = benchSeed(i);
	World& w = worlds[i];
	worldInit(w, *pk, seed);
	int px, pz;
	spreadCandidate(T->sets[set].spread, (uint64_t)seed, 0, 0, px, pz);
	int st = jigsawStructureAt(w, *pk, *T, *J, set, px, pz);
	Piece* pc = pieces + (size_t)i * GPU_MAX_PIECES;
	int m = st < 0 ? -2 : jigsawLayout(w, *pk, *J, st, px, pz, pc, GPU_MAX_PIECES);
	out[i] = m;
	catOut[i] = m > 0 ? countCategory(*J, pc, m, cat) : 0;
}

// GPU vs CPU: same piece counts / category counts; returns mismatches
inline int gpuCheck(const Pack& P, const StructTables& T, const JigsawHost& H, int set, int n) {
	const JigsawTables& J = *H.t;
	int cat = H.categoryIndex("small_house");
	JCK(cudaDeviceSetLimit(cudaLimitStackSize, 16384));
	PackDev* dPk;
	StructTables* dT;
	JigsawTables* dJ;
	World* dW;
	Piece* dP;
	int *dOut, *dCat;
	JCK(cudaMalloc(&dPk, sizeof(PackDev)));
	JCK(cudaMalloc(&dT, sizeof(StructTables)));
	JCK(cudaMalloc(&dJ, sizeof(JigsawTables)));
	JCK(cudaMalloc(&dW, sizeof(World) * n));
	JCK(cudaMalloc(&dP, sizeof(Piece) * GPU_MAX_PIECES * n));
	JCK(cudaMalloc(&dOut, sizeof(int) * n));
	JCK(cudaMalloc(&dCat, sizeof(int) * n));
	JCK(cudaMemcpy(dPk, P.dev.get(), sizeof(PackDev), cudaMemcpyHostToDevice));
	JCK(cudaMemcpy(dT, &T, sizeof T, cudaMemcpyHostToDevice));
	JCK(cudaMemcpy(dJ, &J, sizeof J, cudaMemcpyHostToDevice));
	kJigsawLayout<<<(n + 63) / 64, 64>>>(dPk, dT, dJ, set, n, dW, dP, dOut, dCat, cat);  // warm-up
	JCK(cudaDeviceSynchronize());
	double t0 = now();
	kJigsawLayout<<<(n + 63) / 64, 64>>>(dPk, dT, dJ, set, n, dW, dP, dOut, dCat, cat);
	JCK(cudaGetLastError());
	JCK(cudaDeviceSynchronize());
	double t1 = now();
	std::vector<int> out(n), cats(n);
	JCK(cudaMemcpy(out.data(), dOut, sizeof(int) * n, cudaMemcpyDeviceToHost));
	JCK(cudaMemcpy(cats.data(), dCat, sizeof(int) * n, cudaMemcpyDeviceToHost));
	cudaFree(dPk); cudaFree(dT); cudaFree(dJ); cudaFree(dW); cudaFree(dP); cudaFree(dOut); cudaFree(dCat);
	World* w = new World();
	std::vector<Piece> buf(GPU_MAX_PIECES);
	int bad = 0, villages = 0;
	for (int i = 0; i < n; i++) {
		int64_t seed = benchSeed(i);
		worldInit(*w, *P.dev, seed);
		int px, pz;
		spreadCandidate(T.sets[set].spread, (uint64_t)seed, 0, 0, px, pz);
		int st = jigsawStructureAt(*w, *P.dev, T, J, set, px, pz);
		int m = st < 0 ? -2 : jigsawLayout(*w, *P.dev, J, st, px, pz, buf.data(), GPU_MAX_PIECES);
		int c = m > 0 ? countCategory(J, buf.data(), m, cat) : 0;
		villages += m > 0;
		if (m != out[i] || c != cats[i]) {
			if (bad++ < 5) printf("  seed %lld: cpu %d pieces (%d small houses), gpu %d (%d)\n", (long long)seed, m, c, out[i], cats[i]);
		}
	}
	delete w;
	printf("gpu: %d seeds (%d villages) in %.1f ms incl. worldInit + start check (%.2f us/seed); cpu/gpu mismatches: %d\n", n, villages,
	       (t1 - t0) * 1e3, (t1 - t0) / n * 1e6, bad);
	return bad;
}
#endif

}  // namespace jcli

inline int jigsawCommand(const Pack& P, const StructTables& T, int argc, char** argv) {
	using namespace jcli;
	std::string cmd = argv[2];
	JigsawHost H;
	try {
		H = buildJigsaw(P);
	} catch (std::exception& e) {
		fprintf(stderr, "%s\n", e.what());
		return 2;
	}
	if (!H.t) {
		fprintf(stderr, "%s\n", H.error.c_str());
		return 2;
	}
	const JigsawTables& J = *H.t;
	const PackDev& pk = *P.dev;

	if (cmd == "village-categories") {  // building categories each jigsaw structure can contain
		for (size_t st = 0; st < P.structures.size(); st++) {
			if (st >= MAX_STRUCTS || J.structOf[st] < 0) continue;
			for (auto& c : jigsawCategories(H, (int)st)) printf("piece %s %s\n", shortName(P.structures[st].name).c_str(), c.c_str());
		}
		return 0;
	}

	std::vector<Piece> buf(CLI_MAX_PIECES);
	if (cmd == "village") {
		if (argc < 6) {
			fprintf(stderr, "usage: seedwright <pack> village <seed> <blockX> <blockZ> [--heights villages-file]\n");
			return 2;
		}
		int64_t seed = std::stoll(argv[3]);
		int bx = std::stoi(argv[4]), bz = std::stoi(argv[5]);
		int cx = bx >> 4, cz = bz >> 4;
		std::vector<OracleStart> oracle;
		const HeightGrid* grid = nullptr;
		HeightGrid hg;
		for (int i = 6; i + 1 < argc; i++)
			if (std::string(argv[i]) == "--heights") {
				int64_t os;
				if (!readOracle(argv[i + 1], os, oracle)) {
					fprintf(stderr, "cannot read %s\n", argv[i + 1]);
					return 2;
				}
				for (auto& s : oracle)
					if (s.cx == cx && s.cz == cz) {
						hg = s.grid();
						grid = &hg;
					}
				if (!grid) fprintf(stderr, "(no oracle start at chunk %d %d in %s: using the height estimate)\n", cx, cz, argv[i + 1]);
			}
		World* w = new World();
		worldInit(*w, pk, seed);
		int found = 0;
		for (int si = 0; si < T.nsets; si++) {
			bool jig = false;
			for (int k = 0; k < T.sets[si].nstructs; k++) jig |= J.structOf[T.sets[si].structs[k]] >= 0;
			if (!jig || T.sets[si].type != SET_SPREAD || !isStructureChunk(T, si, (uint64_t)seed, cx, cz)) continue;
			int stOld = structureAt(*w, pk, T, si, cx, cz);
			int st = jigsawStructureAt(*w, pk, T, J, si, cx, cz, grid);
			printf("set %s chunk %d %d: start %s (chunk-middle check: %s)\n", P.sets[si].name.c_str(), cx, cz,
			       st >= 0 ? P.structures[st].name.c_str() : "none", stOld >= 0 ? P.structures[stOld].name.c_str() : "none");
			if (st < 0) continue;
			int n = jigsawLayout(*w, pk, J, st, cx, cz, buf.data(), (int)buf.size(), grid);
			if (n < 0) {
				printf("  more than %d pieces\n", (int)buf.size());
				continue;
			}
			found++;
			printf("pieces %d\n", n);
			for (int i = 0; i < n; i++) printf("  %3d d%d %s\n", i, buf[i].depth, pieceDesc(P, buf[i], cx << 4, cz << 4).c_str());
			printf("buildings:");
			for (auto& kv : categoryCounts(H, buf.data(), n)) printf(" %s=%d", kv.first.c_str(), kv.second);
			printf("\n");
		}
		if (!found) printf("no jigsaw structure starts at chunk %d %d\n", cx, cz);
		delete w;
		return 0;
	}

	if (cmd == "village-verify") {
		// per start: structure choice, piece list with the estimated heights, piece list with vanilla's heights
		int tot = 0, stOk = 0, exactEst = 0, catsEst = 0, exactTrue = 0, catsTrue = 0, countEst = 0, xzEst = 0, xzTrue = 0;
		long long catPairs = 0, catPairsOk = 0, bVan = 0, bEst = 0, bAbs = 0;
		std::map<std::string, std::array<int, 2>> catDiff;  // category -> (villages where we have more, fewer)
		long long dh = 0, dh2 = 0, nh = 0, nhBad = 0;
		std::map<std::string, std::array<int, 3>> perType;  // starts, exact (est), exact (true heights)
		bool verbose = getenv("SW_VERBOSE") != nullptr;
		Terrain TR;  // exact heights (26.3+ packs)
		bool haveTerrain = TR.load(argv[1]);
		int exactTer = 0, catsTer = 0;
		long long terCols = 0;
		double terTime = 0;
		int dumpRadius = getenv("SW_RADIUS") ? atoi(getenv("SW_RADIUS")) : 100, candTot = 0, candJig = 0, candOld = 0;  // oracle villages=R
		for (int f = 3; f < argc; f++) {
			int64_t seed;
			std::vector<OracleStart> starts;
			if (!readOracle(argv[f], seed, starts)) {
				fprintf(stderr, "cannot read %s\n", argv[f]);
				return 2;
			}
			World* w = new World();
			worldInit(*w, pk, seed);
			std::unique_ptr<TerrainWorld> tw(haveTerrain ? new TerrainWorld(TR, seed) : nullptr);
			printf("== %s (seed %lld): %zu starts\n", argv[f], (long long)seed, starts.size());
			// start selection over every candidate of the dumped sets (the dump covers all candidates within R)
			{
				std::map<std::pair<int, int>, std::string> van;
				for (auto& s : starts) van[{s.cx, s.cz}] = s.structure;
				for (int si = 0; si < T.nsets; si++) {
					if (P.sets[si].name != "villages" && P.sets[si].name != "pillager_outposts") continue;
					for (int cx = -dumpRadius; cx <= dumpRadius; cx++)
						for (int cz = -dumpRadius; cz <= dumpRadius; cz++) {
							if (!isStructureChunk(T, si, (uint64_t)seed, cx, cz)) continue;
							auto v = van.find({cx, cz});
							std::string want = v == van.end() ? "-" : v->second;
							int a = jigsawStructureAt(*w, pk, T, J, si, cx, cz), b = structureAt(*w, pk, T, si, cx, cz);
							candTot++;
							candJig += (a >= 0 ? P.structures[a].name : "-") == want;
							candOld += (b >= 0 ? P.structures[b].name : "-") == want;
						}
				}
			}
			for (auto& s : starts) {
				auto it = P.structureIndex.find(s.structure);
				if (it == P.structureIndex.end()) continue;
				int st = it->second, si = setOf(P, st);
				tot++;
				HeightGrid g = s.grid();
				// height estimate quality on this footprint
				for (int k = 0; k < s.hnz; k += 4)
					for (int i = 0; i < s.hnx; i += 4) {
						int h = s.heights[(size_t)k * s.hnx + i];
						if (h == HEIGHT_UNKNOWN) continue;
						int e = estimateSurfaceY(*w, pk, s.hx0 + i, s.hz0 + k);
						dh += e - h;
						dh2 += (long long)(e - h) * (e - h);
						nh++;
						nhBad += std::abs(e - h) > 4;
					}
				int pick = si >= 0 ? jigsawStructureAt(*w, pk, T, J, si, s.cx, s.cz, &g) : -1;
				bool pickOk = pick == st;
				stOk += pickOk;
				std::multiset<std::string> want;
				for (size_t i = 0; i < s.elem.size(); i++) {
					char b[128];
					snprintf(b, sizeof b, " %d %d %d %d %d %d %d", s.rot[i], s.box[i][0], s.box[i][1], s.box[i][2], s.box[i][3], s.box[i][4], s.box[i][5]);
					want.insert(s.elem[i] + b);
				}
				auto wantCats = categoryCounts(s);
				std::multiset<std::string> wantXZ;  // ignoring y (heights only move pieces vertically)
				for (size_t i = 0; i < s.elem.size(); i++)
					wantXZ.insert(s.elem[i] + " " + std::to_string(s.rot[i]) + " " + std::to_string(s.box[i][0]) + " " + std::to_string(s.box[i][2]) + " " +
					              std::to_string(s.box[i][3]) + " " + std::to_string(s.box[i][5]));
				auto& pt = perType[s.structure];
				pt[0]++;
				if (tw) {  // exact terrain: the whole start (structure choice + layout) from our own heights
					HeightGrid eg = terrainGrid(*tw);
					long long c0 = tw->columns();
					double t0 = now();
					int pick2 = si >= 0 ? jigsawStructureAt(*w, pk, T, J, si, s.cx, s.cz, &eg) : -1;
					int n = pick2 == st ? jigsawLayout(*w, pk, J, st, s.cx, s.cz, buf.data(), (int)buf.size(), &eg) : -1;
					terTime += now() - t0;
					terCols += tw->columns() - c0;
					std::multiset<std::string> got;
					for (int i = 0; i < n; i++) got.insert(pieceDesc(P, buf[i], s.cx << 4, s.cz << 4));
					bool exact = n >= 0 && got == want;
					exactTer += exact;
					catsTer += n > 0 && categoryCounts(H, buf.data(), n) == wantCats;
					if (!exact) printf("  MISMATCH %s %d %d with exact terrain: %d pieces vs vanilla %zu\n", s.structure.c_str(), s.cx, s.cz, n, s.elem.size());
				}
				for (int mode = 0; mode < 2; mode++) {
					int n = jigsawLayout(*w, pk, J, st, s.cx, s.cz, buf.data(), (int)buf.size(), mode ? &g : nullptr);
					std::multiset<std::string> got;
					for (int i = 0; i < n; i++) got.insert(pieceDesc(P, buf[i], s.cx << 4, s.cz << 4));
					bool exact = got == want;
					bool cats = n > 0 && categoryCounts(H, buf.data(), n) == wantCats;
					std::multiset<std::string> gotXZ;
					for (int i = 0; i < n; i++)
						gotXZ.insert(elementName(P, buf[i].elem) + " " + std::to_string(buf[i].rot) + " " + std::to_string(buf[i].x0 + (s.cx << 4)) + " " +
						             std::to_string(buf[i].z0 + (s.cz << 4)) + " " + std::to_string(buf[i].x1 + (s.cx << 4)) + " " +
						             std::to_string(buf[i].z1 + (s.cz << 4)));
					if (mode == 1) xzTrue += gotXZ == wantXZ;
					if (mode == 0) {
						xzEst += gotXZ == wantXZ;
						auto gc = categoryCounts(H, buf.data(), n);
						for (int c : H.structCategories[J.structOf[st]]) {
							const std::string& cn = H.categories[c];
							int a = gc.count(cn) ? gc[cn] : 0, b = wantCats.count(cn) ? wantCats.at(cn) : 0;
							catPairs++;
							catPairsOk += a == b;
							bEst += a;
							bVan += b;
							bAbs += std::abs(a - b);
							if (a > b) catDiff[cn][0]++;
							if (a < b) catDiff[cn][1]++;
						}
						exactEst += exact;
						catsEst += cats;
						countEst += n == (int)s.elem.size();
						pt[1] += exact;
					} else {
						exactTrue += exact;
						catsTrue += cats;
						pt[2] += exact;
					}
					if (!exact && (mode == 1 || verbose)) {
						printf("  %s %s %d %d: %s heights: %d pieces vs vanilla %zu%s\n", mode ? "MISMATCH" : "diff", s.structure.c_str(), s.cx, s.cz,
						       mode ? "vanilla" : "estimated", n, s.elem.size(), pickOk ? "" : " (structure choice differs too)");
						if (mode == 1 || verbose) {
							// first differing piece in vanilla order
							for (int i = 0; i < n && i < (int)s.elem.size(); i++) {
								std::string a = pieceDesc(P, buf[i], s.cx << 4, s.cz << 4);
								char b[128];
								snprintf(b, sizeof b, " %d %d %d %d %d %d %d", s.rot[i], s.box[i][0], s.box[i][1], s.box[i][2], s.box[i][3], s.box[i][4],
								         s.box[i][5]);
								if (a != s.elem[i] + b) {
									printf("    first difference at piece %d: got %s | want %s%s\n", i, a.c_str(), s.elem[i].c_str(), b);
									break;
								}
							}
						}
					}
					if (mode == 0 && verbose) {
						printf("    buildings est:");
						for (auto& kv : categoryCounts(H, buf.data(), n)) printf(" %s=%d", kv.first.c_str(), kv.second);
						printf("\n    buildings van:");
						for (auto& kv : wantCats) printf(" %s=%d", kv.first.c_str(), kv.second);
						printf("\n");
					}
				}
			}
			delete w;
		}
		printf("candidates %d (villages + outposts within %d chunks): start/no-start + structure right %d (jigsawStructureAt, est. heights), %d "
		       "(structureAt, chunk middle)\n",
		       candTot, dumpRadius, candJig, candOld);
		printf("starts %d: structure choice (exact xz, real heights) %d/%d\n", tot, stOk, tot);
		printf("  with vanilla heights: identical piece lists %d/%d, identical ignoring y %d/%d, identical building counts %d/%d\n", exactTrue, tot,
		       xzTrue, tot, catsTrue, tot);
		if (haveTerrain)
			printf("  with exact terrain (terrain.h): identical piece lists %d/%d, identical building counts %d/%d (%.1f columns, %.1f ms per start)\n",
			       exactTer, tot, catsTer, tot, (double)terCols / std::max(1, tot), terTime * 1e3 / std::max(1, tot));
		printf("  with estimated heights: identical piece lists %d/%d, identical ignoring y %d/%d, identical building counts %d/%d, same piece "
		       "count %d/%d\n",
		       exactEst, tot, xzEst, tot, catsEst, tot, countEst, tot);
		if (catPairs)
			printf("  with estimated heights: per (start, category) count agreement %lld/%lld (%.1f%%); buildings %lld ours vs %lld vanilla, "
			       "sum |diff| %lld\n",
			       catPairsOk, catPairs, 100.0 * catPairsOk / catPairs, bEst, bVan, bAbs);
		if (verbose)
			for (auto& kv : catDiff) printf("    %-18s more in %d starts, fewer in %d\n", kv.first.c_str(), kv.second[0], kv.second[1]);
		for (auto& kv : perType)
			printf("    %-28s %3d starts: exact %3d (est. heights) %3d (vanilla heights)\n", kv.first.c_str(), kv.second[0], kv.second[1], kv.second[2]);
		if (nh)
			printf("  height estimate on piece footprints: mean error %+.2f, rms %.2f, |err| > 4 in %.1f%% of %lld columns\n", (double)dh / nh,
			       sqrt((double)dh2 / nh), 100.0 * nhBad / nh, nh);
		return exactTrue == tot && (!haveTerrain || exactTer == tot) ? 0 : 1;
	}

	if (cmd == "village-heights") {  // exact terrain (terrain.h) vs the oracle's heights on every dumped column
		Terrain TR;
		if (!TR.load(argv[1])) {
			fprintf(stderr, "%s\n", TR.error.c_str());
			return 2;
		}
		int step = getenv("SW_STEP") ? atoi(getenv("SW_STEP")) : 1;
		long long n = 0, bad = 0;
		double t0 = now();
		for (int f = 3; f < argc; f++) {
			int64_t seed;
			std::vector<OracleStart> starts;
			if (!readOracle(argv[f], seed, starts)) {
				fprintf(stderr, "cannot read %s\n", argv[f]);
				return 2;
			}
			TerrainWorld tw(TR, seed);
			long long fb = 0, fn = 0;
			for (auto& s : starts)
				for (int k = 0; k < s.hnz; k += step)
					for (int i = 0; i < s.hnx; i += step) {
						int h = s.heights[(size_t)k * s.hnx + i];
						if (h == HEIGHT_UNKNOWN) continue;
						int e = tw.height(s.hx0 + i, s.hz0 + k);
						fn++;
						if (e != h) {
							if (fb++ < 5) printf("  seed %lld column %d %d: ours %d vanilla %d\n", (long long)seed, s.hx0 + i, s.hz0 + k, e, h);
						}
					}
			printf("%s: %lld/%lld columns exact\n", argv[f], fn - fb, fn);
			n += fn;
			bad += fb;
		}
		double t1 = now();
		printf("heights: %lld/%lld exact (%.3f ms per column)\n", n - bad, n, (t1 - t0) * 1e3 / std::max(1LL, n));
		return bad ? 1 : 0;
	}

	if (cmd == "village-bench") {  // single thread: villages around the origin of N seeds; --gpu M: GPU run checked against CPU
		int N = argc > 3 ? atoi(argv[3]) : 50;
		for (int i = 3; i + 1 < argc; i++)
			if (std::string(argv[i]) == "--gpu") {
#ifndef SW_CPU_ONLY
				for (size_t k = 0; k < P.sets.size(); k++)
					if (P.sets[k].name == "villages") return gpuCheck(P, T, H, (int)k, atoi(argv[i + 1])) ? 1 : 0;
				return 2;
#else
				fprintf(stderr, "CPU-only build\n");
				return 2;
#endif
			}
		int vs = -1;
		for (size_t i = 0; i < P.sets.size(); i++)
			if (P.sets[i].name == "villages") vs = (int)i;
		if (vs < 0) return 2;
		World* w = new World();
		double tStart = 0, tLayout = 0;
		long long nStart = 0, nVillage = 0, nPieces = 0, maxPieces = 0;
		uint64_t hash = 1469598103934665603ULL;
		std::map<std::string, int> cats;
		for (int i = 0; i < N; i++) {
			int64_t seed = (int64_t)mixStafford13(i + 777);
			worldInit(*w, pk, seed);
			for (int rx = -2; rx <= 1; rx++)
				for (int rz = -2; rz <= 1; rz++) {
					int px, pz;
					spreadCandidate(T.sets[vs].spread, (uint64_t)seed, rx, rz, px, pz);
					double a = now();
					int st = jigsawStructureAt(*w, pk, T, J, vs, px, pz);
					double b = now();
					tStart += b - a;
					nStart++;
					if (st < 0) continue;
					int n = jigsawLayout(*w, pk, J, st, px, pz, buf.data(), (int)buf.size());
					tLayout += now() - b;
					nVillage++;
					nPieces += n;
					for (int k = 0; k < n; k++) {  // FNV-1a over the piece lists (regression check)
						const Piece& q = buf[k];
						int16_t f[10] = {q.elem, (int16_t)q.rot, q.x0, q.y0, q.z0, q.x1, q.y1, q.z1, (int16_t)q.depth, q.region};
						for (int16_t v : f) hash = (hash ^ (uint16_t)v) * 1099511628211ULL;
					}
					maxPieces = std::max<long long>(maxPieces, n);
				}
		}
		printf("%lld candidates: jigsawStructureAt %.1f us each; %lld villages: layout %.1f us each, %.1f pieces avg, max %lld (hash %016llx)\n",
		       nStart, tStart / nStart * 1e6, nVillage, tLayout / std::max(1LL, nVillage) * 1e6, (double)nPieces / std::max(1LL, nVillage), maxPieces,
		       (unsigned long long)hash);
		delete w;
		return 0;
	}
	fprintf(stderr, "unknown command %s\n", cmd.c_str());
	return 2;
}

}  // namespace sw

// seedwright CLI. Builds as CUDA (nvcc) or CPU-only (g++ -x c++ -DSW_CPU_ONLY).
#include "pack.h"
#include "structures.h"
#include "query.h"
#include "query_parse.h"
#include "jigsaw_cli.h"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <set>
#include <atomic>
#include <mutex>
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
		int climBad[6] = {0}, biomeBad = 0, chainBad = 0, chainLeaf = -1;
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
			// vanilla's RTree starts from the thread's previous result: replay the oracle's query order
			chainLeaf = leafFromClimate(pk, c, chainLeaf);
			chainBad += P.biomes[pk.rnodes[chainLeaf].biome] != biome;
		}
		printf("  points: %d  climate mismatches t/h/c/e/d/w = %d %d %d %d %d %d  biome mismatches = %d (fresh search), %d (replaying vanilla's lastResult chain)\n",
		       n, climBad[0], climBad[1], climBad[2], climBad[3], climBad[4], climBad[5], biomeBad, chainBad);
		bad += chainBad;
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
		if (getenv("SW_SPAWN_DEBUG")) {
			int64_t bestTheirs = INT64_MAX;
			for (int x = ox * 16; x < ox * 16 + 16; x++)
				for (int z = oz * 16; z < oz * 16 + 16; z++) bestTheirs = std::min(bestTheirs, spawnFitness(*w, pk, x, z));
			printf("  spawn debug: ours (%d,%d) fitness %lld; best in vanilla's chunk %lld; origin %lld\n", sx, sz, (long long)spawnFitness(*w, pk, sx, sz),
			       (long long)bestTheirs, (long long)spawnFitness(*w, pk, 0, 0));
		}
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


// ------------------------------------------------------------------ search

static bool nameMatches(const std::string& full, const std::string& want) {
	std::string a = full.substr(full.find(':') + 1);
	return a == want || a.find(want) != std::string::npos;
}

// village layouts (jigsaw tables), built once; null host tables when the pack predates them
static JigsawHost* gJig = nullptr;
static const JigsawHost& jig(const Pack& P) {
	if (!gJig) {
		gJig = new JigsawHost();
		try {
			*gJig = buildJigsaw(P);
		} catch (std::exception& e) {
			gJig->t.reset();
			gJig->error = e.what();
		}
	}
	return *gJig;
}

static QueryNames queryNames(const Pack& P) {
	QueryNames N;
	N.P = &P;
	const JigsawHost& H = jig(P);
	if (H.t) {
		N.buildings.resize(P.structures.size());
		for (size_t st = 0; st < P.structures.size() && st < 64; st++)
			for (auto& name : jigsawCategories(H, (int)st)) N.buildings[st].push_back({name, H.categoryIndex(name)});
	}
	return N;
}

// Old-style flags (--need/--biome/--stronghold/--spawn-within) and --where expressions both build
// children of one top-level ALL node, so they can be mixed.
static Query parseQuery(const Pack& P, const StructTables& T, std::vector<std::string>& args, std::map<std::string, std::string>& opt, std::string* jsonOut = nullptr) {
	Query q;
	memset(&q, 0, sizeof q);
	q.anchorSpawn = 1;
	q.slack = 256;
	QueryNames N = queryNames(P);
	QueryParser qp(N, q);
	q.root = qp.newNode(N_ALL);
	for (size_t i = 0; i < args.size(); i++) {
		std::string a = args[i];
		auto val = [&]() {
			if (i + 1 >= args.size()) throw std::runtime_error("missing value for " + a);
			return args[++i];
		};
		auto at = [](const std::string& v, std::string& head, int& r) {
			size_t p = v.find('@');
			if (p == std::string::npos) throw std::runtime_error("expected NAME@RADIUS in " + v);
			head = v.substr(0, p);
			r = std::stoi(v.substr(p + 1));
			if (r < 0) throw std::runtime_error("negative distance in " + v);
		};
		if (a == "--where") {
			qp.parseInto(q.root, val());
		} else if (a == "--need") {
			int ni = qp.newNode(N_STRUCTURE);
			Node& c = q.n[ni];
			std::string head;
			at(val(), head, c.radius);
			size_t colon = head.find(':');
			c.count = colon == std::string::npos ? 1 : std::stoi(head.substr(colon + 1));
			if (c.count < 1) throw std::runtime_error("count must be >= 1 in " + head);
			std::string name = head.substr(0, colon);
			bool exact = !name.empty() && name[0] == '=';  // "=NAME": that one structure, no set/substring match
			if (exact) name = name.substr(1);
			for (int si = 0; si < T.nsets; si++) {
				bool setMatch = !exact && (P.sets[si].name == name || P.sets[si].name == name + "s");
				for (int st : P.sets[si].structs)
					if (setMatch || (exact ? P.structures[st].name.substr(P.structures[st].name.find(':') + 1) == name
					                       : nameMatches(P.structures[st].name, name))) {
						c.structMask |= 1ULL << st;
						c.setMask |= 1u << si;
					}
			}
			if (!c.structMask) throw std::runtime_error("no structure matches '" + name + "'");
			bool isSet = false;
			for (auto& set : P.sets) isSet |= set.name == name || set.name == name + "s";
			qp.label[ni] = exact || isSet ? name : "*" + name + "*";  // same meaning in the query language
			qp.attach(q.root, ni);
		} else if (a == "--biome") {  // NAME@RADIUS[@Y|@any]
			int ni = qp.newNode(N_BIOME);
			Node& c = q.n[ni];
			std::string v = val(), head;
			size_t p2 = v.find('@', v.find('@') + 1);
			c.y = 64;
			if (p2 != std::string::npos) {
				std::string y = v.substr(p2 + 1);
				c.y = y == "any" ? (int)Y_ANY : std::stoi(y);
				v = v.substr(0, p2);
			}
			at(v, head, c.radius);
			c.biome = P.biome(head);
			if (c.biome < 0) throw std::runtime_error("unknown biome " + head);
			qp.attach(q.root, ni);
		} else if (a == "--stronghold") {
			int ni = qp.newNode(N_STRONGHOLD);
			Node& c = q.n[ni];
			std::string head;
			at(val(), head, c.radius);
			c.count = head.empty() ? 1 : std::stoi(head);
			if (c.count < 1 || c.count > MAX_STRONGHOLDS) throw std::runtime_error("stronghold count must be 1.." + std::to_string(MAX_STRONGHOLDS));
			qp.attach(q.root, ni);
		} else if (a == "--spawn-within") {
			int ni = qp.newNode(N_SPAWN);
			q.n[ni].radius = std::stoi(val());
			q.needSpawn = 1;
			qp.attach(q.root, ni);
		} else if (a == "--origin") {
			q.anchorSpawn = 0;
		} else if (a == "--slack") {
			q.slack = std::stoi(val());
		} else if (a.rfind("--", 0) == 0) {
			opt[a.substr(2)] = (i + 1 < args.size() && args[i + 1].rfind("--", 0) != 0) ? val() : "1";
		} else {
			throw std::runtime_error("unexpected argument " + a);
		}
	}
	q.exact = opt.count("seeds") ? 1 : 0;
	if (q.n[q.root].first < 0) throw std::runtime_error("no conditions (use --where '...' or --need/--biome/...)");
	if (jsonOut) *jsonOut = queryJson(P, q, q.root, N, qp.label);
	else fprintf(stderr, "query: %s\n", queryText(P, q, q.root, N, qp.label).c_str());
	return q;
}

static std::string hitName(const Pack& P, int what) {
	if (what >= 0) return P.structures[what].name.substr(10);
	if (what == WHAT_STRONGHOLD) return "stronghold";
	if (what <= WHAT_BUILDING0) {
		int cat = WHAT_BUILDING0 - what;
		return gJig && cat < (int)gJig->categories.size() ? buildingKey(gJig->categories[cat]) : "building";
	}
	return P.biomes[-1 - what].substr(10);
}

// "<seed>  spawn X Z  | name x z | >name x z ..." ('>' per nesting level: found inside the item before)
// "123", "0x7b" or "2^N" (the GUI writes powers of two that way)
static uint64_t parseU64(const std::string& v) {
	size_t c = v.find('^');
	if (c == std::string::npos) return std::stoull(v, nullptr, 0);
	uint64_t b = std::stoull(v.substr(0, c)), e = std::stoull(v.substr(c + 1)), r = 1;
	while (e--) r *= b;
	return r;
}

static void printHit(const Pack& P, const Hit& h) {
	printf("%lld  spawn %d %d ", (long long)h.seed, h.spawnX, h.spawnZ);
	for (int i = 0; i < h.n; i++) printf(" | %s%s %d %d", std::string(h.depth[i], '>').c_str(), hitName(P, h.what[i]).c_str(), h.x[i], h.z[i]);
	printf("\n");
	fflush(stdout);
}

static std::vector<int64_t> parseSeeds(const std::string& s) {
	std::vector<int64_t> v;
	std::stringstream ss(s);
	std::string t;
	while (std::getline(ss, t, ',')) v.push_back(std::stoll(t));
	return v;
}

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static int searchCpu(const Pack& P, const StructTables& T, const Query& q, std::map<std::string, std::string>& opt) {
	const JigsawTables* J = jig(P).t.get();
	const PackDev& pk = *P.dev;
	long long maxResults = opt.count("max") ? std::stoll(opt["max"]) : 20;
	std::atomic<long long> found{0};
	if (opt.count("seeds")) {
		for (int64_t s : parseSeeds(opt["seeds"])) {
			auto* w = new World();
			worldInit(*w, pk, s);
			Hit h;
			std::vector<Piece> pcs(q.hasBuildings ? MAX_PIECES : 0);
			bool ok = stageB(*w, pk, T, q, h, J, pcs.data());  // q.exact: no stage A / slack prefilter for given seeds
			if (ok && opt.count("check-stage-a") && !stageA(T, q, (uint64_t)s)) printf("STAGE-A-UNSOUND %lld\n", (long long)s);  // test aid
			if (ok) printHit(P, h);
			else printf("%lld  no match\n", (long long)s);
			delete w;
		}
		return 0;
	}
	uint64_t start = opt.count("start") ? parseU64(opt["start"]) : 0;
	uint64_t count = opt.count("count") ? parseU64(opt["count"]) : (1ULL << 24);
	// stage A streams in chunks so a huge --count never holds all survivors at once
	std::mutex mu;
	const uint64_t chunkA = 1ULL << 24;
	long long tested = 0;
	uint64_t survivors = 0;
	double tA = 0, tB = 0;
	for (uint64_t off = 0; off < count && found < maxResults; off += chunkA) {
		uint64_t n = std::min(chunkA, count - off);
		double t0 = now();
		std::vector<uint64_t> lows;
#pragma omp parallel
		{
			std::vector<uint64_t> mine;
#pragma omp for schedule(dynamic, 65536)
			for (long long i = 0; i < (long long)n; i++) {
				uint64_t lo = (start + off + i) & 0xFFFFFFFFFFFFULL;
				if (stageA(T, q, lo)) mine.push_back(lo);
			}
			std::lock_guard<std::mutex> g(mu);
			lows.insert(lows.end(), mine.begin(), mine.end());
		}
		std::sort(lows.begin(), lows.end());
		survivors += lows.size();
		double t1 = now();
		tA += t1 - t0;
		for (uint64_t lo : lows) {
			if (found >= maxResults) break;
#pragma omp parallel
			{
				World* w = new World();
				std::vector<Piece> pcs(q.hasBuildings ? MAX_PIECES : 0);
#pragma omp for schedule(dynamic, 64)
				for (int hi = 0; hi < 65536; hi++) {
					if (found >= maxResults) continue;
					int64_t s = (int64_t)(((uint64_t)hi << 48) | lo);
					worldInit(*w, pk, s);
					Hit h;
					if (stageB(*w, pk, T, q, h, J, pcs.data()) && found++ < maxResults) {
						std::lock_guard<std::mutex> g(mu);
						printHit(P, h);
					}
				}
				delete w;
			}
			tested += 65536;
			tB = now() - t1;
			fprintf(stderr, "\rstage B: %lld full seeds, %.0f seeds/s, %lld results   ", tested, tested / std::max(1e-9, tB), std::min((long long)found, maxResults));
		}
		fprintf(stderr, "\nstage A: %llu of %llu low-48 seeds done (%.1f M/s), %llu survivors so far\n", (unsigned long long)std::min(off + n, count),
		        (unsigned long long)count, (off + n) / std::max(1e-9, tA) / 1e6, (unsigned long long)survivors);
	}
	if (!survivors) fprintf(stderr, "hint: no seed in this range can place the structures this close (counts too high or distances too small for how far apart they spawn)\n");
	fprintf(stderr, "done: %lld full seeds tested, %lld results\n", tested, std::min((long long)found, maxResults));
	return 0;
}

#ifndef SW_CPU_ONLY
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

__global__ void kStageA(const StructTables* T, const Query* q, uint64_t start, uint64_t count, uint64_t* out, unsigned long long* nout, unsigned long long cap) {
	for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < count; i += (uint64_t)gridDim.x * blockDim.x) {
		uint64_t lo = (start + i) & 0xFFFFFFFFFFFFULL;
		if (stageA(*T, *q, lo)) {
			unsigned long long k = atomicAdd(nout, 1ull);  // 64-bit: a 2^32 batch where everything survives must not wrap to 0
			if (k < cap) out[k] = lo;
		}
	}
}

__global__ void kStageB(const PackDev* pk, const StructTables* T, const Query* q, const int64_t* seeds, unsigned n, World* worlds, Hit* hits, unsigned* nhits, unsigned cap,
                        const JigsawTables* J, Piece* pieces) {
	World& w = worlds[blockIdx.x * blockDim.x + threadIdx.x];
	for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
		worldInit(w, *pk, seeds[i]);
		Hit h;
		if (stageB(w, *pk, *T, *q, h, J, pieces ? pieces + (size_t)(blockIdx.x * blockDim.x + threadIdx.x) * MAX_PIECES : nullptr)) {
			unsigned k = atomicAdd(nhits, 1u);
			if (k < cap) hits[k] = h;
		}
	}
}

static int searchGpu(const Pack& P, const StructTables& T, const Query& q, std::map<std::string, std::string>& opt) {
	long long maxResults = opt.count("max") ? std::stoll(opt["max"]) : 20;
	CK(cudaDeviceSetLimit(cudaLimitStackSize, q.hasBuildings ? 24576 : 16384));
	PackDev* dPk;
	StructTables* dT;
	Query* dQ;
	CK(cudaMalloc(&dPk, sizeof(PackDev)));
	CK(cudaMalloc(&dT, sizeof(StructTables)));
	CK(cudaMalloc(&dQ, sizeof(Query)));
	CK(cudaMemcpy(dPk, P.dev.get(), sizeof(PackDev), cudaMemcpyHostToDevice));
	CK(cudaMemcpy(dT, &T, sizeof T, cudaMemcpyHostToDevice));
	CK(cudaMemcpy(dQ, &q, sizeof q, cudaMemcpyHostToDevice));
	int sms;
	CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
	const int tpb = 128, blocksB = sms * 4;
	const unsigned nWorlds = blocksB * tpb;
	World* dW;
	CK(cudaMalloc(&dW, sizeof(World) * nWorlds));
	// jigsaw tables (structure starts checked where vanilla does) and, for building queries, layout buffers
	JigsawTables* dJ = nullptr;
	Piece* dPieces = nullptr;
	if (const JigsawTables* J = jig(P).t.get()) {
		CK(cudaMalloc(&dJ, sizeof(JigsawTables)));
		CK(cudaMemcpy(dJ, J, sizeof(JigsawTables), cudaMemcpyHostToDevice));
		if (q.hasBuildings) CK(cudaMalloc(&dPieces, sizeof(Piece) * MAX_PIECES * (size_t)nWorlds));
	}
	const unsigned long long capA = 1ull << 22;
	const unsigned capH = 4096;
	uint64_t* dLows;
	unsigned long long* dN;
	unsigned* dNH;
	Hit* dHits;
	CK(cudaMalloc(&dLows, sizeof(uint64_t) * capA));
	CK(cudaMalloc(&dN, 8));
	CK(cudaMalloc(&dNH, 4));
	CK(cudaMalloc(&dHits, sizeof(Hit) * capH));
	int64_t* dSeeds;
	const unsigned batch = 1u << 18;
	CK(cudaMalloc(&dSeeds, sizeof(int64_t) * batch));
	std::vector<int64_t> hseeds;
	long long found = 0;
	auto runB = [&](const std::vector<int64_t>& seeds) {
		for (size_t off = 0; off < seeds.size() && found < maxResults; off += batch) {
			unsigned n = (unsigned)std::min<size_t>(batch, seeds.size() - off);
			CK(cudaMemcpy(dSeeds, seeds.data() + off, sizeof(int64_t) * n, cudaMemcpyHostToDevice));
			CK(cudaMemset(dNH, 0, 4));
			kStageB<<<blocksB, tpb>>>(dPk, dT, dQ, dSeeds, n, dW, dHits, dNH, capH, dJ, dPieces);
			CK(cudaGetLastError());
			unsigned nh;
			CK(cudaMemcpy(&nh, dNH, 4, cudaMemcpyDeviceToHost));
			nh = std::min(nh, capH);
			std::vector<Hit> hits(nh);
			CK(cudaMemcpy(hits.data(), dHits, sizeof(Hit) * nh, cudaMemcpyDeviceToHost));
			std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) { return a.seed < b.seed; });
			for (auto& h : hits)
				if (found++ < maxResults) printHit(P, h);
		}
	};
	if (opt.count("seeds")) {
		auto v = parseSeeds(opt["seeds"]);
		runB(v);
		return 0;
	}
	uint64_t start = opt.count("start") ? parseU64(opt["start"]) : 0;
	uint64_t count = opt.count("count") ? parseU64(opt["count"]) : (1ULL << 32);
	// Stage A in batches; a batch with more survivors than the buffer holds is shrunk and redone, so no
	// survivor is ever dropped and seeds are covered in order.
	uint64_t chunkA = 1ULL << 24;  // grows x4 while batches stay well under the buffer
	double tA = 0, tB = 0;
	long long tested = 0;
	unsigned long long survivors = 0;
	for (uint64_t off = 0; off < count && found < maxResults;) {
		uint64_t n = std::min(chunkA, count - off);
		double t0 = now();
		CK(cudaMemset(dN, 0, 8));
		kStageA<<<sms * 16, 256>>>(dT, dQ, start + off, n, dLows, dN, capA);
		CK(cudaGetLastError());
		unsigned long long nl;
		CK(cudaMemcpy(&nl, dN, 8, cudaMemcpyDeviceToHost));
		tA += now() - t0;
		if (nl > capA) {
			chunkA = std::max<uint64_t>(capA, n / (nl / capA + 1) / 2);  // capA seeds can never overflow
			continue;
		}
		std::vector<uint64_t> lows(nl);
		CK(cudaMemcpy(lows.data(), dLows, 8ull * nl, cudaMemcpyDeviceToHost));
		std::sort(lows.begin(), lows.end());
		survivors += nl;
		off += n;
		if (nl < capA / 8 && n == chunkA) chunkA = std::min<uint64_t>(chunkA * 4, 1ULL << 32);
		fprintf(stderr, "stage A: %llu of %llu low-48 seeds done (%.0f M/s), %llu survivors so far\n", (unsigned long long)off,
		        (unsigned long long)count, off / std::max(tA, 1e-9) / 1e6, survivors);
		for (size_t i = 0; i < lows.size() && found < maxResults;) {
			hseeds.clear();
			for (; i < lows.size() && hseeds.size() + 65536 <= batch; i++)
				for (int hi = 0; hi < 65536; hi++) hseeds.push_back((int64_t)(((uint64_t)hi << 48) | lows[i]));
			double tb = now();
			runB(hseeds);
			CK(cudaDeviceSynchronize());
			tB += now() - tb;
			tested += hseeds.size();
			fprintf(stderr, "\rstage B: %lld full seeds, %.0f seeds/s, %lld results   ", tested, tested / std::max(tB, 1e-9), std::min(found, maxResults));
		}
		if (tested) fprintf(stderr, "\n");
	}
	if (!survivors) fprintf(stderr, "hint: no seed in this range can place the structures this close (counts too high or distances too small for how far apart they spawn)\n");
	fprintf(stderr, "done: %lld full seeds tested, %lld results\n", tested, std::min(found, maxResults));
	return 0;
}
#endif

int main(int argc, char** argv) {
	if (argc < 3) {
		fprintf(stderr, "usage: seedwright <pack> verify <oracle files...>\n       seedwright <pack> search ...\n       seedwright <pack> list\n       seedwright <pack> parse '<query>'\n");
		return 2;
	}
	Pack P = loadPack(argv[1]);
	StructTables T = buildTables(P);
	std::string cmd = argv[2];
	if (cmd == "verify") return verify(P, T, std::vector<std::string>(argv + 3, argv + argc));
	if (cmd.rfind("village", 0) == 0) return jigsawCommand(P, T, argc, argv);  // village, village-categories, village-verify, village-bench
	if (cmd == "list") {  // machine-readable names for front ends: "set NAME", "structure SET NAME", "biome NAME"
		for (auto& s : P.sets) {
			printf("set %s\n", s.name.c_str());
			for (int st : s.structs) printf("structure %s %s\n", s.name.c_str(), P.structures[st].name.substr(P.structures[st].name.find(':') + 1).c_str());
		}
		for (auto& b : P.biomes) printf("biome %s\n", b.substr(b.find(':') + 1).c_str());
		// "building STRUCTURE NAME" (canonical names, see buildingKey), approximate layouts
		QueryNames N = queryNames(P);
		for (size_t st = 0; st < N.buildings.size(); st++) {
			std::vector<std::string> keys;
			for (auto& b : N.buildings[st]) {
				std::string k = buildingKey(b.first);
				if (std::find(keys.begin(), keys.end(), k) == keys.end()) keys.push_back(k);
			}
			std::sort(keys.begin(), keys.end());
			for (auto& k : keys) printf("building %s %s\n", P.structures[st].name.substr(P.structures[st].name.find(':') + 1).c_str(), k.c_str());
		}
		return 0;
	}
	if (cmd == "bench") {  // single-thread per-stage cost
		const PackDev& pk = *P.dev;
		World* w = new World();
		int N = argc > 3 ? atoi(argv[3]) : 300;
		int vil = setIndex(P, "villages");
		double t0 = now(), tInit = 0, tSpawn = 0, tStruct = 0, tBiome = 0;
		long long nStruct = 0, sink = 0;
		for (int i = 0; i < N; i++) {
			int64_t seed = (int64_t)mixStafford13(i + 1);
			double a = now();
			worldInit(*w, pk, seed);
			double b = now();
			int sx, sz;
			spawnOrigin(*w, pk, sx, sz);
			double c = now();
			for (int rx = -1; rx <= 0; rx++)
				for (int rz = -1; rz <= 0; rz++) {
					int px, pz;
					spreadCandidate(T.sets[vil].spread, (uint64_t)seed, rx, rz, px, pz);
					sink += structureAt(*w, pk, T, vil, px, pz);
					nStruct++;
				}
			double d = now();
			for (int k = 0; k < 64; k++) sink += biomeAt(*w, pk, k * 37, 16, k * 11, 0);
			double e = now();
			tInit += b - a; tSpawn += c - b; tStruct += d - c; tBiome += e - d;
		}
		printf("per seed: init %.1f us, spawn %.1f us, village structureAt %.1f us each, biomeAt %.2f us each (sink %lld, total %.2fs)\n",
		       tInit / N * 1e6, tSpawn / N * 1e6, tStruct / nStruct * 1e6, tBiome / N / 64 * 1e6, sink, now() - t0);
		return 0;
	}
	if (cmd == "parse") {  // seedwright <pack> parse '<query>' [more search flags]: JSON tree on stdout, errors on stderr
		std::vector<std::string> args;
		for (int i = 3; i < argc; i++) {
			if (i == 3 && std::string(argv[i]).rfind("--", 0) != 0) args.push_back("--where");
			args.push_back(argv[i]);
		}
		std::map<std::string, std::string> opt;
		std::string json;
		try {
			parseQuery(P, T, args, opt, &json);
		} catch (std::exception& e) {
			fprintf(stderr, "%s\n", e.what());
			return 2;
		}
		printf("%s\n", json.c_str());
		return 0;
	}
	if (cmd == "search") {
		std::vector<std::string> args(argv + 3, argv + argc);
		std::map<std::string, std::string> opt;
		Query q;
		try {
			q = parseQuery(P, T, args, opt);
		} catch (std::exception& e) {
			fprintf(stderr, "%s\n", e.what());
			return 2;
		}
#ifndef SW_CPU_ONLY
		if (!opt.count("cpu")) return searchGpu(P, T, q, opt);
#endif
		return searchCpu(P, T, q, opt);
	}
	fprintf(stderr, "unknown command %s\n", cmd.c_str());
	return 2;
}

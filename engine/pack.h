// Host-only: loads a version pack (tools/extract.py output) into PackDev + host-side structure tables.
#pragma once
#include "sw_core.h"
#include <openssl/evp.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw {

inline void md5pair(const std::string& s, uint64_t& lo, uint64_t& hi) {
	unsigned char d[16];
	unsigned int n = 0;
	EVP_Digest(s.data(), s.size(), d, &n, EVP_md5(), nullptr);
	lo = hi = 0;
	for (int i = 0; i < 8; i++) lo = (lo << 8) | d[i];
	for (int i = 8; i < 16; i++) hi = (hi << 8) | d[i];
}

inline float bitsF(const std::string& h) {
	uint32_t u = (uint32_t)std::stoul(h, nullptr, 16);
	float f;
	memcpy(&f, &u, 4);
	return f;
}
inline double bitsD(const std::string& h) {
	uint64_t u = std::stoull(h, nullptr, 16);
	double d;
	memcpy(&d, &u, 8);
	return d;
}

struct StructureInfo {
	std::string name, type, project, y;
	std::vector<int> biomes;  // biome indices
	std::vector<uint8_t> biomeMask;
};

struct StructureSetInfo {
	std::string name, type, freqMethod, exclusionSet;
	int salt = 0, exclusionChunks = 0, exclusionIndex = -1;
	float frequency = 1.0f;
	Spread spread{};
	int distance = 0, ringSpread = 0, count = 0;  // concentric rings
	std::vector<uint8_t> preferredMask;
	std::vector<int> structs, weights;
};

// Jigsaw data (pack section "jigsaw", appended by tools/extract.py; absent in older packs). Indices refer to
// the vectors below; -1 = none / non-existent pool. See engine/jigsaw.h for the device tables.
struct JigsawJigInfo {
	int x, y, z, front, top, joint;  // template-local pos; Direction index (down up north south west east); joint 0 rollable 1 aligned
	int name, target, pool, placement, selection;
};
struct JigsawTemplateInfo {
	std::string name;
	int sx = 0, sy = 0, sz = 0;
	std::vector<JigsawJigInfo> jig;  // palette order (y, x, z)
};
struct JigsawElementInfo {
	std::string kind, proj, arg;  // kind single|legacy|feature|list|empty; arg = template index / feature id
	std::vector<int> subs;  // list members (element indices)
};
struct JigsawPoolInfo {
	std::string name;
	int fallback = -1, maxSize = 0;
	std::vector<std::pair<int, int>> entries;  // (element, weight)
};
struct JigsawStructureInfo {
	std::string name, heightKind, project;
	int startPool = -1, startJigsaw = -1, size = 0, maxH = 0, maxV = 0, h0 = 0, h1 = 0, expansion = 0, padBottom = 0, padTop = 0, aliases = 0;
};
struct JigsawPackInfo {
	bool present = false;
	std::string featureName;  // FeaturePoolElement jigsaw name ("*" = null, matches any target: 26.3+)
	std::vector<std::string> names;
	std::vector<JigsawTemplateInfo> templates;
	std::vector<JigsawElementInfo> elements;
	std::vector<JigsawPoolInfo> pools;
	std::vector<JigsawStructureInfo> structures;
};

struct Pack {
	std::string version;
	JigsawPackInfo jigsaw;
	std::vector<std::string> biomes;
	std::map<std::string, int> biomeIndex;
	std::vector<StructureInfo> structures;
	std::map<std::string, int> structureIndex;
	std::vector<StructureSetInfo> sets;
	std::unique_ptr<PackDev> dev{new PackDev()};

	int biome(const std::string& n) const {
		auto it = biomeIndex.find(n.find(':') == std::string::npos ? "minecraft:" + n : n);
		return it == biomeIndex.end() ? -1 : it->second;
	}
};

// ---- RTree build: port of Climate.RTree.build (stable sorts reproduce Java's List.sort) ----
struct BNode {
	int64_t lo[7], hi[7];
	int biome = -1;
	std::vector<BNode*> kids;
};

inline void spanOf(BNode* n) {
	for (int d = 0; d < 7; d++) {
		n->lo[d] = INT64_MAX;
		n->hi[d] = INT64_MIN;
		for (BNode* k : n->kids) {
			n->lo[d] = std::min(n->lo[d], k->lo[d]);
			n->hi[d] = std::max(n->hi[d], k->hi[d]);
		}
	}
}

inline BNode* rtreeBuild(std::vector<BNode*> children, std::vector<std::unique_ptr<BNode>>& pool, int per) {
	const int dims = 7;
	if (children.size() == 1) return children[0];
	auto center = [](BNode* n, int d) { return (n->lo[d] + n->hi[d]) / 2; };
	if ((int)children.size() <= per) {
		std::stable_sort(children.begin(), children.end(), [&](BNode* a, BNode* b) {
			int64_t ma = 0, mb = 0;
			for (int d = 0; d < dims; d++) {
				ma += std::llabs(center(a, d));
				mb += std::llabs(center(b, d));
			}
			return ma < mb;
		});
		pool.emplace_back(new BNode());
		BNode* n = pool.back().get();
		n->kids = children;
		spanOf(n);
		return n;
	}
	auto sortBy = [&](std::vector<BNode*>& v, int dim, bool absolute) {
		std::stable_sort(v.begin(), v.end(), [&](BNode* a, BNode* b) {
			for (int k = 0; k < dims; k++) {
				int d = (dim + k) % dims;
				int64_t ca = center(a, d), cb = center(b, d);
				if (absolute) {
					ca = std::llabs(ca);
					cb = std::llabs(cb);
				}
				if (ca != cb) return ca < cb;
			}
			return false;
		});
	};
	auto bucketize = [&](const std::vector<BNode*>& nodes) {
		std::vector<std::vector<BNode*>> buckets;
		std::vector<BNode*> cur;
		int expected = (int)std::pow((double)per, std::floor(std::log((double)nodes.size() - 0.01) / std::log((double)per)));
		for (BNode* c : nodes) {
			cur.push_back(c);
			if ((int)cur.size() >= expected) {
				buckets.push_back(cur);
				cur.clear();
			}
		}
		if (!cur.empty()) buckets.push_back(cur);
		return buckets;
	};
	auto span = [&](const std::vector<BNode*>& b, int64_t* lo, int64_t* hi) {
		for (int d = 0; d < dims; d++) {
			lo[d] = INT64_MAX;
			hi[d] = INT64_MIN;
			for (BNode* k : b) {
				lo[d] = std::min(lo[d], k->lo[d]);
				hi[d] = std::max(hi[d], k->hi[d]);
			}
		}
	};
	int64_t minCost = INT64_MAX;
	int minDim = -1;
	std::vector<std::vector<BNode*>> best;
	for (int d = 0; d < dims; d++) {
		sortBy(children, d, false);
		auto buckets = bucketize(children);
		int64_t total = 0;
		for (auto& b : buckets) {
			int64_t lo[7], hi[7];
			span(b, lo, hi);
			for (int k = 0; k < dims; k++) total += std::llabs(hi[k] - lo[k]);
		}
		if (minCost > total) {
			minCost = total;
			minDim = d;
			best = buckets;
		}
	}
	// SubTrees for the buckets, sorted by |center| along minDim (then the following dims)
	std::vector<BNode*> subs;
	for (auto& b : best) {
		pool.emplace_back(new BNode());
		BNode* t = pool.back().get();
		t->kids = b;
		spanOf(t);
		subs.push_back(t);
	}
	sortBy(subs, minDim, true);
	pool.emplace_back(new BNode());
	BNode* n = pool.back().get();
	for (BNode* t : subs) n->kids.push_back(rtreeBuild(t->kids, pool, per));
	spanOf(n);
	return n;
}

inline void flatten(BNode* root, PackDev& pk) {
	// BFS so that each node's children are contiguous
	std::vector<BNode*> order{root};
	std::map<BNode*, int> idx{{root, 0}};
	for (size_t i = 0; i < order.size(); i++)
		for (BNode* k : order[i]->kids) {
			idx[k] = (int)order.size();
			order.push_back(k);
		}
	if (order.size() > MAX_RNODES) throw std::runtime_error("rtree too large");
	pk.nrnodes = (int)order.size();
	for (size_t i = 0; i < order.size(); i++) {
		RNode& r = pk.rnodes[i];
		BNode* b = order[i];
		memcpy(r.lo, b->lo, sizeof r.lo);
		memcpy(r.hi, b->hi, sizeof r.hi);
		r.biome = b->biome;
		r.nchild = (int)b->kids.size();
		r.child0 = r.nchild ? idx[b->kids[0]] : -1;
	}
}

inline Pack loadPack(const std::string& path) {
	std::ifstream in(path);
	if (!in) throw std::runtime_error("cannot open pack " + path);
	Pack P;
	PackDev& pk = *P.dev;
	memset(&pk, 0, sizeof pk);
	std::string line, tok;
	auto next = [&]() {
		if (!std::getline(in, line)) throw std::runtime_error("pack truncated");
		return std::istringstream(line);
	};
	auto header = [&](const char* want) {
		auto ss = next();
		std::string k;
		int n;
		ss >> k >> n;
		if (k != want) throw std::runtime_error(std::string("pack: expected ") + want + " got " + k);
		return n;
	};
	{
		auto ss = next();
		std::string magic, engine, spawn;
		int v;
		ss >> magic >> v >> P.version >> engine >> spawn;
		pk.legacy = engine == "legacy";
		pk.oldSpawn = spawn == "oldspawn";
	}
	int nb = header("biomes");
	for (int i = 0; i < nb; i++) {
		std::getline(in, line);
		P.biomeIndex[line] = i;
		P.biomes.push_back(line);
	}
	int np = header("params");
	std::vector<std::unique_ptr<BNode>> pool;
	std::vector<BNode*> leaves;
	for (int i = 0; i < np; i++) {
		auto ss = next();
		pool.emplace_back(new BNode());
		BNode* b = pool.back().get();
		ss >> b->biome;
		for (int d = 0; d < 6; d++) ss >> b->lo[d] >> b->hi[d];
		ss >> b->lo[6];
		b->hi[6] = b->lo[6];
		leaves.push_back(b);
	}
	flatten(rtreeBuild(leaves, pool, pk.legacy ? 6 : 19), pk);  // Climate.RTree CHILDREN_PER_NODE: 6 before 26.3, 19 since
	{
		auto ss = next();
		std::string k;
		ss >> k;
		for (int i = 0; i < 6; i++) ss >> pk.climate[i];
	}
	{
		auto ss = next();
		std::string k;
		ss >> k >> pk.minY >> pk.height >> pk.seaLevel;
	}
	int nn = header("noises");
	pk.nnoises = nn;
	for (int n = 0; n < nn; n++) {
		auto ss = next();
		std::string name;
		int nl;
		ss >> name;
		NoiseDesc& nd = pk.noises[n];
		md5pair(name, nd.hlo, nd.hhi);
		nd.layer0 = pk.nlayers;
		if (!(ss >> nl)) {  // "name legacy firstOctave count amps..." (PerlinNoise + legacy NormalNoise)
			ss.clear();
			std::string tag;
			int first, cnt;
			ss >> tag >> first >> cnt;
			std::vector<double> amps(cnt);
			for (auto& a : amps) {
				ss >> tok;
				a = bitsD(tok);
			}
			double inF = std::pow(2.0, (double)first), valF = std::pow(2.0, (double)(cnt - 1)) / (std::pow(2.0, (double)cnt) - 1.0);
			int lo = INT32_MAX, hi = INT32_MIN;
			for (int i = 0; i < cnt; i++) {
				if (amps[i] != 0.0) {
					lo = std::min(lo, i);
					hi = std::max(hi, i);
					for (int second = 0; second < 2; second++) {
						LayerDesc& L = pk.layers[pk.nlayers++];
						L.freq = inF;
						L.ampD = amps[i];
						L.vf = valF;
						L.second = second;
						md5pair("octave_" + std::to_string(first + i), L.hlo, L.hhi);
					}
				}
				inF *= 2.0;
				valF /= 2.0;
			}
			nd.valueFactor = 0.16666666666666666 / (0.1 * (1.0 + 1.0 / (double)(hi - lo + 1)));
			nd.nlayers = pk.nlayers - nd.layer0;
			continue;
		}
		for (int l = 0; l < nl; l++) {
			int oct;
			std::string fr, am;
			ss >> oct >> fr >> am;
			for (int second = 0; second < 2; second++) {
				LayerDesc& L = pk.layers[pk.nlayers++];
				L.freq = second ? bitsD(fr) * 1.0181268882175227 : bitsD(fr);
				L.amp = bitsF(am);
				L.second = second;
				md5pair("octave_" + std::to_string(oct), L.hlo, L.hhi);
			}
		}
		nd.nlayers = pk.nlayers - nd.layer0;
	}
	int no = header("ops");
	pk.nops = no;
	static const std::map<std::string, OpCode> codes = {
		{"const", OP_CONST}, {"add", OP_ADD}, {"mul", OP_MUL}, {"min", OP_MIN}, {"max", OP_MAX}, {"abs", OP_ABS},
		{"square", OP_SQUARE}, {"cube", OP_CUBE}, {"half_negative", OP_HALFNEG}, {"quarter_negative", OP_QUARTNEG},
		{"squeeze", OP_SQUEEZE}, {"invert", OP_INVERT}, {"clamp", OP_CLAMP}, {"lerp", OP_LERP}, {"ygrad", OP_YGRAD},
		{"shift_a", OP_SHIFTA}, {"shift_b", OP_SHIFTB}, {"noise", OP_NOISE}, {"range_choice", OP_RANGE}, {"spline", OP_SPLINE}};
	for (int i = 0; i < no; i++) {
		auto ss = next();
		Op& o = pk.ops[i];
		std::string name, bar;
		int cached, na, nf, nd, ni;
		ss >> name >> cached >> na;
		o.code = codes.at(name);
		o.cached = (uint8_t)cached;
		int16_t* args[3] = {&o.a, &o.b, &o.c};
		o.a = o.b = o.c = -1;
		for (int k = 0; k < na; k++) {
			int v;
			ss >> v;
			*args[k] = (int16_t)v;
		}
		ss >> bar >> nf;
		float fp[2] = {0, 0};
		for (int k = 0; k < nf; k++) {
			ss >> tok;
			fp[k] = bitsF(tok);
		}
		ss >> bar >> nd;
		double dp[2] = {0, 0};
		for (int k = 0; k < nd; k++) {
			ss >> tok;
			dp[k] = bitsD(tok);
		}
		ss >> bar >> ni;
		int ip[3] = {0, 0, 0};
		for (int k = 0; k < ni; k++) ss >> ip[k];
		o.f0 = fp[0];
		o.f1 = fp[1];
		o.d0 = dp[0];
		o.d1 = dp[1];
		o.i0 = ip[0];
		o.i1 = ip[1];
		if (o.code == OP_YGRAD) {  // GradientFunction.compileSampler
			int from = ip[0], to = ip[1];
			o.i0 = from;
			o.i1 = std::min(from, to);
			o.i2 = std::max(from, to);
			o.f0 = fp[0];
			o.f1 = (fp[1] - fp[0]) / (float)(to - from);
			o.i3 = to;
		}
		if (o.code == OP_NOISE) {
			auto isZero = [&](int16_t a) { return pk.ops[a].code == OP_CONST && pk.ops[a].f0 == 0.0f; };
			o.unshifted = isZero(o.a) && isZero(o.b) && isZero(o.c);
		}
	}
	// y-dependence (for column evaluation): gradients, noises with a y scale, and anything downstream
	auto ydepOf = [&](int i) { return i >= 0 && pk.ops[i].ydep; };
	for (int i = 0; i < no; i++) {
		Op& o = pk.ops[i];
		o.ydep = o.code == OP_YGRAD || (o.code == OP_NOISE && o.d1 != 0.0) || ydepOf(o.a) || ydepOf(o.b) || ydepOf(o.c);
	}
	int ns = header("splines");
	pk.nsplines = ns;
	int npts = 0;
	for (int i = 0; i < ns; i++) {
		auto ss = next();
		std::string k;
		ss >> k;
		SplineNode& sn = pk.splines[i];
		if (k == "c") {
			ss >> tok;
			sn.coord = -1;
			sn.value = bitsF(tok);
		} else {
			ss >> sn.coord >> sn.n;
			sn.start = npts;
			for (int j = 0; j < sn.n; j++) {
				std::string l, d;
				ss >> l >> d >> pk.schild[npts];
				pk.sloc[npts] = bitsF(l);
				pk.sder[npts] = bitsF(d);
				npts++;
			}
		}
	}
	// splines: y-dependent if any coordinate is
	for (int pass = 0; pass < 2; pass++)
		for (int i = 0; i < no; i++) {
			Op& o = pk.ops[i];
			if (o.code != OP_SPLINE) continue;
			for (int j = 0; j < ns; j++)
				if (pk.splines[j].coord >= 0 && pk.ops[pk.splines[j].coord].ydep) o.ydep = 1;  // conservative
			for (int k = i + 1; k < no; k++) {
				Op& u = pk.ops[k];
				u.ydep |= ydepOf(u.a) || ydepOf(u.b) || ydepOf(u.c);
			}
		}
	int nsp = header("spawn_target");
	pk.nspawn = nsp;
	for (int p = 0; p < nsp; p++) {
		auto ss = next();
		std::string k;
		ss >> k;
		if (k == "fixed") {  // 1.21-style ParameterPoint: t,h,c,e,d,w ranges + offset; target depth is forced to 0
			int64_t r[13];
			for (auto& x : r) ss >> x;
			int map[5] = {0, 1, 2, 3, 5};  // depth term: |0 - [d]| is constant; vanilla's is the point 0 -> 0
			pk.spawnTerms[p] = 5;
			for (int t = 0; t < 5; t++) {
				pk.spawnOp[p][t] = pk.climate[map[t]];
				pk.spawnLo[p][t] = r[map[t] * 2];
				pk.spawnHi[p][t] = r[map[t] * 2 + 1];
			}
			continue;
		}
		ss >> pk.spawnTerms[p];
		for (int t = 0; t < pk.spawnTerms[p]; t++) ss >> pk.spawnOp[p][t] >> pk.spawnLo[p][t] >> pk.spawnHi[p][t];
	}
	spawnPrepare(pk);
	int nset = header("structure_sets");
	std::vector<std::vector<std::string>> setTokens;
	for (int i = 0; i < nset; i++) {
		auto ss = next();
		std::vector<std::string> t;
		while (ss >> tok) t.push_back(tok);
		setTokens.push_back(t);
	}
	int nst = header("structures");
	for (int i = 0; i < nst; i++) {
		auto ss = next();
		StructureInfo si;
		int n;
		ss >> si.name >> si.type >> si.project >> si.y >> n;
		si.biomeMask.assign(nb, 0);
		for (int k = 0; k < n; k++) {
			ss >> tok;
			auto it = P.biomeIndex.find(tok);
			if (it != P.biomeIndex.end()) {
				si.biomes.push_back(it->second);
				si.biomeMask[it->second] = 1;
			}
		}
		P.structureIndex[si.name] = (int)P.structures.size();
		P.structures.push_back(si);
	}
	for (auto& t : setTokens) {
		StructureSetInfo s;
		size_t k = 0;
		s.name = t[k++];
		s.type = t[k++];
		s.salt = std::stoi(t[k++]);
		s.frequency = bitsF(t[k++]);
		s.freqMethod = t[k++];
		s.exclusionSet = t[k++];
		s.exclusionChunks = std::stoi(t[k++]);
		auto readStructs = [&]() {
			int n = std::stoi(t[k++]);
			for (int j = 0; j < n; j++) {
				s.structs.push_back(P.structureIndex.at(t[k++]));
				s.weights.push_back(std::stoi(t[k++]));
			}
		};
		if (s.type == "random_spread") {
			s.spread.spacing = std::stoi(t[k++]);
			s.spread.separation = std::stoi(t[k++]);
			s.spread.triangular = t[k++] == "triangular";
			readStructs();
		} else if (s.type == "concentric_rings") {
			s.distance = std::stoi(t[k++]);
			s.ringSpread = std::stoi(t[k++]);
			s.count = std::stoi(t[k++]);
			readStructs();
			int n = std::stoi(t[k++]);
			s.preferredMask.assign(nb, 0);
			for (int j = 0; j < n; j++) {
				auto it = P.biomeIndex.find(t[k++]);
				if (it != P.biomeIndex.end()) s.preferredMask[it->second] = 1;
			}
		} else {
			readStructs();
		}
		s.spread.salt = s.salt;
		s.spread.frequency = s.frequency;
		s.spread.freqMethod = s.freqMethod == "legacy_type_1" ? 1 : s.freqMethod == "legacy_type_2" ? 2 : s.freqMethod == "legacy_type_3" ? 3 : 0;
		P.sets.push_back(s);
	}
	// only sets with a structure whose biomes occur in the overworld parameter list (hasBiomesForStructureSet)
	std::vector<StructureSetInfo> kept;
	for (auto& s : P.sets) {
		bool any = false;
		for (int st : s.structs) any |= !P.structures[st].biomes.empty();
		if (any) kept.push_back(s);
	}
	P.sets = kept;
	for (auto& s : P.sets)
		for (size_t j = 0; j < P.sets.size(); j++)
			if ("minecraft:" + s.exclusionSet.substr(s.exclusionSet.find(':') + 1) == "minecraft:" + P.sets[j].name) s.exclusionIndex = (int)j;
	// optional trailing jigsaw section (packs from before it existed simply end here)
	if (std::getline(in, line) && line.rfind("jigsaw ", 0) == 0) {
		JigsawPackInfo& J = P.jigsaw;
		{
			std::istringstream ss(line);
			std::string k;
			int ver;
			ss >> k >> ver >> J.featureName;
		}
		int n = header("jigsaw_names");
		for (int i = 0; i < n; i++) {
			std::getline(in, line);
			J.names.push_back(line);
		}
		n = header("jigsaw_templates");
		for (int i = 0; i < n; i++) {
			auto ss = next();
			JigsawTemplateInfo t;
			int nj;
			ss >> t.name >> t.sx >> t.sy >> t.sz >> nj;
			for (int j = 0; j < nj; j++) {
				JigsawJigInfo g;
				ss >> g.x >> g.y >> g.z >> g.front >> g.top >> g.joint >> g.name >> g.target >> g.pool >> g.placement >> g.selection;
				t.jig.push_back(g);
			}
			J.templates.push_back(t);
		}
		n = header("jigsaw_elements");
		for (int i = 0; i < n; i++) {
			auto ss = next();
			JigsawElementInfo e;
			int ns;
			ss >> e.kind >> e.proj >> e.arg >> ns;
			e.subs.resize(ns);
			for (auto& x : e.subs) ss >> x;
			J.elements.push_back(e);
		}
		n = header("jigsaw_pools");
		for (int i = 0; i < n; i++) {
			auto ss = next();
			JigsawPoolInfo p;
			int ne;
			ss >> p.name >> p.fallback >> p.maxSize >> ne;
			p.entries.resize(ne);
			for (auto& e : p.entries) ss >> e.first >> e.second;
			J.pools.push_back(p);
		}
		n = header("jigsaw_structures");
		for (int i = 0; i < n; i++) {
			auto ss = next();
			JigsawStructureInfo s;
			ss >> s.name >> s.startPool >> s.startJigsaw >> s.size >> s.maxH >> s.maxV >> s.heightKind >> s.h0 >> s.h1 >> s.project >> s.expansion >>
				s.padBottom >> s.padTop >> s.aliases;
			J.structures.push_back(s);
		}
		J.present = true;
	}
	return P;
}

}  // namespace sw

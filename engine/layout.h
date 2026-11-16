// Rare-layout mode: instead of scanning low-48 seeds one by one (stage A), generate only the structure seeds
// that put one required structure where the query wants it, by running the LCG backwards.
//
// For a random_spread set the candidate chunk of region (rx, rz) is region*spacing + (nextInt(lim), nextInt(lim))
// from setSeed(regionSeed), regionSeed = rx*K1 + rz*K2 + seed + salt (all mod 2^48). If the query needs a structure
// within R of 0,0, the region's x offset (or z offset) can only take a few values ("allowed"). Fixing the draw
// v = next(31) to those values (v = lim*t + a, or the a-th block of v for power-of-two lim) and the 17 low state
// bits freely gives every LCG state with that draw; stepping back to the region seed and subtracting the region
// term gives every structure seed with that offset: |allowed|/lim of all 2^48, nothing else.
//
// A requirement "N STRUCTURE within R" whose structure can come from E (set, region) pairs holds in at least N of
// them, so any |E|-N+1 of them ("pivots") cover every seed that satisfies it (pigeonhole); a seed is emitted only
// by the first pivot it satisfies (no duplicates). The plan uses the requirement whose pivots cover the smallest
// share of seeds. Every generated seed still goes through stageA, so results are exactly a full scan's, minus
// seeds whose pivot draw hits nextInt's rejection loop (probability < lim/2^31 per seed).
#pragma once
#include "query.h"
#include <algorithm>
#include <functional>
#include <string>
#include <vector>

namespace sw {

enum { MAX_PIVOTS = 16, MAX_LIM = 64 };

struct LayoutPivot {
	int set, rx, rz, axis;  // axis 0: x offset (first draw), 1: z offset (second draw)
	int lim, pow2bits;      // pow2bits >= 0: lim == 1 << pow2bits (nextInt's top-bits path)
	int nA;
	uint8_t A[MAX_LIM];     // allowed offsets, ascending
	uint64_t allowedMask;   // bit a: offset a allowed
	uint64_t Q;             // draws per allowed offset
	uint64_t begin, size;   // index range in the plan
	int64_t regionTerm;     // rx*K1 + rz*K2 + salt
};

struct LayoutPlan {
	int n;
	LayoutPivot p[MAX_PIVOTS];
	uint64_t total;
	uint64_t minv;  // inverse of the LCG multiplier mod 2^48
	double share;   // total / 2^48
};

static const uint64_t LAYOUT_M = 0x5DEECE66DULL, LAYOUT_MASK = (1ULL << 48) - 1;

HD uint64_t layoutPrev(uint64_t s, uint64_t minv) { return ((s - 0xBULL) * minv) & LAYOUT_MASK; }

// seed for plan index idx; false: not a seed this index stands for (rejection draw, or an earlier pivot emits it)
HD bool layoutSeed(const StructTables& T, const LayoutPlan& L, uint64_t idx, uint64_t& seed) {
	int k = 0;
	while (k + 1 < L.n && idx >= L.p[k + 1].begin) k++;
	const LayoutPivot& p = L.p[k];
	uint64_t local = idx - p.begin;
	uint64_t lo17 = local & 0x1FFFF, j = local >> 17;
	uint64_t a = p.A[j / p.Q], t = j % p.Q;
	uint64_t v = p.pow2bits >= 0 ? (a << (31 - p.pow2bits)) + t : t * (uint64_t)p.lim + a;
	uint64_t s = (v << 17) | lo17;  // state after the pivot draw
	if (p.axis == 1) {
		s = layoutPrev(s, L.minv);  // state after the first (x) draw: it must not have been rejected
		if (p.pow2bits < 0 && (s >> 17) >= p.Q * (uint64_t)p.lim) return false;
	}
	uint64_t s0 = layoutPrev(s, L.minv);
	uint64_t r = (s0 ^ LAYOUT_M) & LAYOUT_MASK;
	seed = (r - (uint64_t)p.regionTerm) & LAYOUT_MASK;
	for (int e = 0; e < k; e++) {  // emitted by an earlier pivot?
		const LayoutPivot& q = L.p[e];
		const Spread& sp = T.sets[q.set].spread;
		int px, pz;
		spreadCandidate(sp, seed, q.rx, q.rz, px, pz);
		int off = q.axis ? pz - q.rz * sp.spacing : px - q.rx * sp.spacing;
		if ((q.allowedMask >> off) & 1) return false;
	}
	return true;
}

// Host: plan from the query's top-level "N STRUCTURE within R" conditions. Returns "" or an error.
inline std::string planLayout(const StructTables& T, const Query& q, LayoutPlan& L, std::string& summary,
                              const std::function<std::string(int)>& setName) {
	L = LayoutPlan{};
	uint64_t x = LAYOUT_M;
	for (int i = 0; i < 6; i++) x *= 2 - LAYOUT_M * x;  // Newton: inverse mod 2^64
	L.minv = x & LAYOUT_MASK;
	std::vector<int> tops;
	const Node& root = q.n[q.root];
	if (root.kind == N_ALL)
		for (int c = root.first; c >= 0; c = q.n[c].next) tops.push_back(c);
	else
		tops.push_back(q.root);
	int slack = q.anchorSpawn ? q.slack : 0;
	double best = 1e300;
	std::vector<LayoutPivot> bestP;
	std::string bestWhy;
	for (int ni : tops) {
		const Node& nd = q.n[ni];
		if (nd.kind != N_STRUCTURE || nd.count < 1) continue;
		int R = nd.radius + slack;
		int64_t R2 = (int64_t)R * R;
		std::vector<std::pair<double, LayoutPivot>> E;
		bool usable = true;
		for (int si = 0; si < T.nsets && usable; si++) {
			if (!((nd.setMask >> si) & 1) || T.sets[si].type != SET_SPREAD) continue;
			const Spread& sp = T.sets[si].spread;
			int lim = sp.spacing - sp.separation;
			int r0x = floorDiv((-R - 8) >> 4, sp.spacing), r1x = floorDiv((R + 8) >> 4, sp.spacing);
			for (int rx = r0x; rx <= r1x && usable; rx++)
				for (int rz = r0x; rz <= r1x && usable; rz++) {
					uint64_t ax = 0, az = 0;
					for (int ox = 0; ox < lim; ox++)
						for (int oz = 0; oz < lim; oz++) {
							int64_t bx = (int64_t)(rx * sp.spacing + ox) * 16 + 8, bz = (int64_t)(rz * sp.spacing + oz) * 16 + 8;
							if (bx * bx + bz * bz <= R2) ax |= 1ULL << ox, az |= 1ULL << oz;
						}
					if (!ax) continue;
					if (sp.triangular || lim > MAX_LIM) {
						usable = false;  // offsets are averages of two draws: can't pivot on them
						break;
					}
					LayoutPivot p{};
					p.set = si, p.rx = rx, p.rz = rz, p.lim = lim;
					p.pow2bits = (lim & (lim - 1)) == 0 ? __builtin_ctz(lim) : -1;
					int nx = __builtin_popcountll(ax), nz = __builtin_popcountll(az);
					p.axis = nz < nx;
					p.allowedMask = p.axis ? az : ax;
					p.nA = 0;
					for (int a = 0; a < lim; a++)
						if ((p.allowedMask >> a) & 1) p.A[p.nA++] = (uint8_t)a;
					p.Q = p.pow2bits >= 0 ? (1ULL << (31 - p.pow2bits)) : (1ULL << 31) / (uint64_t)lim;
					p.size = (uint64_t)p.nA * p.Q << 17;
					p.regionTerm = (int64_t)((uint64_t)((int64_t)rx * 341873128712LL + (int64_t)rz * 132897987541LL) + (uint64_t)(int64_t)sp.salt);
					E.push_back({(double)p.nA / lim, p});
				}
		}
		if (!usable || E.empty()) continue;
		if ((int)E.size() < nd.count) return "impossible: only " + std::to_string(E.size()) + " places within that distance can hold one";
		std::sort(E.begin(), E.end(), [](auto& a, auto& b) { return a.first < b.first; });
		size_t np = E.size() - nd.count + 1;
		if (np > MAX_PIVOTS) continue;
		double share = 0;
		for (size_t i = 0; i < np; i++) share += E[i].first;
		if (share < best) {
			best = share;
			bestP.clear();
			for (size_t i = 0; i < np; i++) bestP.push_back(E[i].second);
			bestWhy = std::to_string(nd.count) + " of " + setName(bestP[0].set) + " within " + std::to_string(nd.radius) + " (" +
			          std::to_string(np) + " pivot region" + (np > 1 ? "s" : "") + " of " + std::to_string(E.size()) + ")";
		}
	}
	if (bestP.empty())
		return "rare-layout mode needs at least one top-level 'N STRUCTURE within R' (not a monument/mansion, which use "
		       "averaged offsets)";
	uint64_t at = 0;
	for (auto& p : bestP) {
		p.begin = at;
		at += p.size;
		L.p[L.n++] = p;
	}
	L.total = at;
	L.share = (double)at / (double)(1ULL << 48);
	char buf[256];
	if (L.share < 1)
		snprintf(buf, sizeof buf, "layout plan: %s -> %.3g%% of all 2^48 structure seeds, %.0fx fewer than a full scan", bestWhy.c_str(),
		         100 * L.share, 1.0 / L.share);
	else
		snprintf(buf, sizeof buf, "layout plan: %s -> %.2fx as many candidates as a full scan (overlapping pivots): no faster", bestWhy.c_str(),
		         L.share);
	summary = buf;
	return "";
}

}  // namespace sw

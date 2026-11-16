// seedwright core: everything here compiles for both host (CPU) and CUDA device.
// Bit-exactness rules: no FMA contraction (-ffp-contract=off / nvcc -fmad=false), IEEE float/double,
// Java integer semantics (two's complement wraparound via uint64_t).
#pragma once
#include <stdint.h>
#include <math.h>

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#define HDN __host__ __device__ inline
#else
#define HD inline
#define HDN inline
#endif

namespace sw {

// ---------------------------------------------------------------- Java RNGs

struct JavaRandom {  // java.util.Random / LegacyRandomSource
	uint64_t s;
	HD void setSeed(uint64_t seed) { s = (seed ^ 0x5DEECE66DULL) & 0xFFFFFFFFFFFFULL; }
	HD int next(int bits) {
		s = (s * 0x5DEECE66DULL + 0xBULL) & 0xFFFFFFFFFFFFULL;
		return (int)(uint32_t)(s >> (48 - bits));  // (int)(newSeed >> 48-bits)
	}
	HD int nextInt() { return next(32); }
	HD int nextInt(int bound) {
		if ((bound & -bound) == bound) return (int)(((int64_t)bound * (int64_t)next(31)) >> 31);
		int bits, val;
		do {
			bits = next(31);
			val = bits % bound;
		} while (bits - val + (bound - 1) < 0);
		return val;
	}
	HD int64_t nextLong() {
		int64_t hi = next(32);
		int64_t lo = next(32);
		return (int64_t)((uint64_t)hi << 32) + lo;
	}
	HD float nextFloat() { return (float)next(24) * 5.9604645E-8F; }
	HD double nextDouble() {
		int64_t c = ((int64_t)next(26) << 27) + next(27);
		return (double)c * (double)1.110223E-16F;
	}
};

HD uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

HD uint64_t mixStafford13(uint64_t z) {
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
	return z ^ (z >> 31);
}

struct Xoro {  // XoroshiroRandomSource
	uint64_t lo, hi;
	HD void init(uint64_t l, uint64_t h) {
		lo = l;
		hi = h;
		if ((lo | hi) == 0) {
			lo = 0x9E3779B97F4A7C15ULL;
			hi = 0x6A09E667F3BCC909ULL;
		}
	}
	HD void seed64(uint64_t seed) {  // upgradeSeedTo128bit (mixed)
		uint64_t l = seed ^ 0x6A09E667F3BCC909ULL;
		uint64_t h = l + 0x9E3779B97F4A7C15ULL;
		init(mixStafford13(l), mixStafford13(h));
	}
	HD uint64_t nextLong() {
		uint64_t s0 = lo, s1 = hi;
		uint64_t r = rotl(s0 + s1, 17) + s0;
		s1 ^= s0;
		lo = rotl(s0, 49) ^ s1 ^ (s1 << 21);
		hi = rotl(s1, 28);
		return r;
	}
	HD int nextInt(int bound) {
		uint64_t r = (uint32_t)nextLong();
		uint64_t m = r * (uint64_t)bound;
		uint64_t frac = m & 0xFFFFFFFFULL;
		if (frac < (uint64_t)bound) {
			uint32_t thresh = (uint32_t)(-(uint32_t)bound) % (uint32_t)bound;
			while (frac < thresh) {
				r = (uint32_t)nextLong();
				m = r * (uint64_t)bound;
				frac = m & 0xFFFFFFFFULL;
			}
		}
		return (int)(m >> 32);
	}
	HD double nextDouble() { return (double)(nextLong() >> 11) * (double)1.110223E-16F; }
	HD float nextFloat() { return (float)(nextLong() >> 40) * 5.9604645E-8F; }
};

// ---------------------------------------------------------------- pack data (POD, device-copyable)

enum OpCode : uint8_t {
	OP_CONST, OP_ADD, OP_MUL, OP_MIN, OP_MAX, OP_ABS, OP_SQUARE, OP_CUBE, OP_HALFNEG, OP_QUARTNEG, OP_SQUEEZE,
	OP_INVERT, OP_CLAMP, OP_LERP, OP_YGRAD, OP_SHIFTA, OP_SHIFTB, OP_NOISE, OP_RANGE, OP_SPLINE
};

struct Op {
	uint8_t code, cached, unshifted, ydep;
	int16_t a, b, c;
	float f0, f1;
	double d0, d1;
	int i0, i1, i2;
};

struct SplineNode {
	int coord;  // op index; -1 = constant
	int n, start;
	float value;
};

struct LayerDesc {
	double freq;
	float amp;
	int second;  // 0 = first, 1 = second (x1.0181268882175227) branch
	uint64_t hlo, hhi;  // md5("octave_N")
};

struct NoiseDesc {
	int nlayers, layer0;
	uint64_t hlo, hhi;  // md5("minecraft:name")
};

struct RNode {
	int64_t lo[7], hi[7];
	int child0, nchild, biome;
};

enum { MAX_OPS = 64, MAX_NOISES = 8, MAX_LAYERS = 64, MAX_SPLINES = 512, MAX_SPLINE_PTS = 1024, MAX_RNODES = 9000 };

struct PackDev {
	int nops, nnoises, nlayers, nsplines, nrnodes;
	int climate[6];
	int minY, height, seaLevel;
	int nspawn, spawnTerms[4], spawnOp[4][8];
	int64_t spawnLo[4][8], spawnHi[4][8];
	Op ops[MAX_OPS];
	NoiseDesc noises[MAX_NOISES];
	LayerDesc layers[MAX_LAYERS];
	SplineNode splines[MAX_SPLINES];
	float sloc[MAX_SPLINE_PTS], sder[MAX_SPLINE_PTS];
	int schild[MAX_SPLINE_PTS];
	RNode rnodes[MAX_RNODES];
};

// ---------------------------------------------------------------- Perlin (26.x float GradientNoise)

struct Perlin {
	double ox, oy, oz;
	uint8_t p[256];
};

HD void perlinInit(Perlin& P, Xoro& r) {
	P.ox = r.nextDouble() * 256.0;
	P.oy = r.nextDouble() * 256.0;
	P.oz = r.nextDouble() * 256.0;
	for (int i = 0; i < 256; i++) P.p[i] = (uint8_t)i;
	for (int i = 0; i < 256; i++) {
		int o = r.nextInt(256 - i);
		uint8_t t = P.p[i];
		P.p[i] = P.p[o + i];
		P.p[o + i] = t;
	}
}

HD double wrapCoord(double x) {
	const double HALF = 16777215.999999998;  // Math.nextDown(1.6777216E7)
	return (x >= -HALF && x < HALF) ? x : x - floor(x / 3.3554432E7 + 0.5) * 3.3554432E7;
}

HD float smoothstep(float x) { return x * x * x * (x * (x * 6.0f - 15.0f) + 10.0f); }
HD float lerpf(float a, float p0, float p1) { return p0 + a * (p1 - p0); }

HD void grad(int h, float& gx, float& gy, float& gz) {
	const int8_t G[16][3] = {{1, 1, 0}, {-1, 1, 0}, {1, -1, 0}, {-1, -1, 0}, {1, 0, 1}, {-1, 0, 1}, {1, 0, -1}, {-1, 0, -1},
	                         {0, 1, 1}, {0, -1, 1}, {0, 1, -1}, {0, -1, -1}, {1, 1, 0}, {0, -1, 1}, {-1, 1, 0}, {0, -1, -1}};
	gx = G[h][0];
	gy = G[h][1];
	gz = G[h][2];
}

HD float corner(int h, float x, float y, float z, bool vol) {
	float gx, gy, gz;
	grad(h & 15, gx, gy, gz);
	if (!vol) return gx * x + gy * y + gz * z;  // ((gx*x + gy*y) + gz*z)
	float xz = gx * x + gz * z;
	return xz + gy * y;
}

HD float perlinGet(const Perlin& P, double X, double Y, double Z, bool vol) {
	double x = wrapCoord(X) + P.ox, y = wrapCoord(Y) + P.oy, z = wrapCoord(Z) + P.oz;
	int fx = (int)floor(x), fy = (int)floor(y), fz = (int)floor(z);
	float rx = (float)(x - fx), ry = (float)(y - fy), rz = (float)(z - fz);
#define PM(v) ((int)P.p[(v) & 0xFF])
	int x0 = PM(fx), x1 = PM(fx + 1);
	int xy00 = PM(x0 + fy), xy01 = PM(x0 + fy + 1), xy10 = PM(x1 + fy), xy11 = PM(x1 + fy + 1);
	float d000 = corner(PM(xy00 + fz), rx, ry, rz, vol);
	float d100 = corner(PM(xy10 + fz), rx - 1.0f, ry, rz, vol);
	float d010 = corner(PM(xy01 + fz), rx, ry - 1.0f, rz, vol);
	float d110 = corner(PM(xy11 + fz), rx - 1.0f, ry - 1.0f, rz, vol);
	float d001 = corner(PM(xy00 + fz + 1), rx, ry, rz - 1.0f, vol);
	float d101 = corner(PM(xy10 + fz + 1), rx - 1.0f, ry, rz - 1.0f, vol);
	float d011 = corner(PM(xy01 + fz + 1), rx, ry - 1.0f, rz - 1.0f, vol);
	float d111 = corner(PM(xy11 + fz + 1), rx - 1.0f, ry - 1.0f, rz - 1.0f, vol);
#undef PM
	float ax = smoothstep(rx), ay = smoothstep(ry), az = smoothstep(rz);
	float l00 = lerpf(ax, d000, d100), l10 = lerpf(ax, d010, d110);
	float l01 = lerpf(ax, d001, d101), l11 = lerpf(ax, d011, d111);
	return lerpf(az, lerpf(ay, l00, l10), lerpf(ay, l01, l11));
}

// ---------------------------------------------------------------- per-seed world (climate noises)

struct World {
	int64_t seed;
	Perlin perlin[MAX_LAYERS];
};

HD void worldInit(World& w, const PackDev& pk, int64_t seed) {
	w.seed = seed;
	Xoro r;
	r.seed64((uint64_t)seed);
	uint64_t posLo = r.nextLong(), posHi = r.nextLong();  // RandomState.random = forkPositional()
	for (int n = 0; n < pk.nnoises; n++) {
		const NoiseDesc& nd = pk.noises[n];
		Xoro nr;
		nr.init(nd.hlo ^ posLo, nd.hhi ^ posHi);
		uint64_t f[4];
		for (int k = 0; k < 4; k++) f[k] = nr.nextLong();  // first.forkPositional, second.forkPositional
		for (int l = 0; l < nd.nlayers; l++) {
			const LayerDesc& L = pk.layers[nd.layer0 + l];
			Xoro lr;
			lr.init(L.hlo ^ f[L.second * 2], L.hhi ^ f[L.second * 2 + 1]);
			perlinInit(w.perlin[nd.layer0 + l], lr);
		}
	}
}

HD float noiseGet(const World& w, const PackDev& pk, int n, double x, double y, double z, bool vol, float ampMul) {
	const NoiseDesc& nd = pk.noises[n];
	float v = 0.0f;
	for (int l = 0; l < nd.nlayers; l++) {
		const LayerDesc& L = pk.layers[nd.layer0 + l];
		float a = ampMul == 1.0f ? L.amp : ampMul * L.amp;
		v += a * perlinGet(w.perlin[nd.layer0 + l], x * L.freq, y * L.freq, z * L.freq, vol);
	}
	return v;
}

// ---------------------------------------------------------------- density program (point + volume semantics)

HDN float splineEval(const PackDev& pk, int s, const float* vals) {
	const SplineNode& sn = pk.splines[s];
	if (sn.coord < 0) return sn.value;
	float input = vals[sn.coord];
	const float* loc = pk.sloc + sn.start;
	const float* der = pk.sder + sn.start;
	const int* ch = pk.schild + sn.start;
	int n = sn.n;
	int from = 0, len = n;  // Mth.binarySearch(0, n, i -> input < loc[i]) - 1
	while (len > 0) {
		int half = len / 2, mid = from + half;
		if (input < loc[mid]) len = half;
		else {
			from = mid + 1;
			len -= half + 1;
		}
	}
	int start = from - 1, last = n - 1;
	if (start < 0) {
		float v = splineEval(pk, ch[0], vals);
		return der[0] == 0.0f ? v : v + der[0] * (input - loc[0]);
	}
	if (start == last) {
		float v = splineEval(pk, ch[last], vals);
		return der[last] == 0.0f ? v : v + der[last] * (input - loc[last]);
	}
	float x1 = loc[start], x2 = loc[start + 1];
	float t = (input - x1) / (x2 - x1);
	float d1 = der[start], d2 = der[start + 1];
	float y1 = splineEval(pk, ch[start], vals), y2 = splineEval(pk, ch[start + 1], vals);
	float a = d1 * (x2 - x1) - (y2 - y1);
	float b = -d2 * (x2 - x1) + (y2 - y1);
	return lerpf(t, y1, y2) + t * (1.0f - t) * lerpf(t, a, b);
}

// Evaluates ops [0, upto] at a block position. vol=true: DensityVolume semantics for every op.
// colHit=true (point mode): cached ops take their volume value (a point query landing inside a
// column volume a cache cell still holds, as in Structure.GenerationContext).
HDN void evalOpsState(const World& w, const PackDev& pk, int bx, int by, int bz, bool vol, bool colHit, int upto, float* pv, float* vv,
                      bool ydepOnly) {
	bool needV = vol || colHit;
	for (int pass = needV ? 0 : 1; pass < (vol ? 1 : 2); pass++) {
		bool V = pass == 0;  // pass 0: volume values, pass 1: point values
		float* R = V ? vv : pv;
		for (int i = 0; i <= upto; i++) {
			const Op& o = pk.ops[i];
			if (ydepOnly && !o.ydep) continue;
			if (!V && colHit && o.cached) {
				pv[i] = vv[i];
				continue;
			}
			float A = o.a >= 0 ? R[o.a] : 0.0f, B = o.b >= 0 ? R[o.b] : 0.0f;
			float r;
			switch (o.code) {
			case OP_CONST: r = o.f0; break;
			case OP_ADD: r = A + B; break;
			case OP_MUL: r = A * B; break;
			case OP_MIN: r = B < A ? B : A; break;
			case OP_MAX: r = B > A ? B : A; break;
			case OP_ABS: r = fabsf(A); break;
			case OP_SQUARE: r = A * A; break;
			case OP_CUBE: r = A * A * A; break;
			case OP_HALFNEG: r = A > 0.0f ? A : A * 0.5f; break;
			case OP_QUARTNEG: r = A > 0.0f ? A : A * 0.25f; break;
			case OP_INVERT: r = 1.0f / A; break;
			case OP_SQUEEZE: {
				float c = A < -1.0f ? -1.0f : (A > 1.0f ? 1.0f : A);
				r = c / 2.0f - c * c * c / 24.0f;
				break;
			}
			case OP_CLAMP: r = A < o.f0 ? o.f0 : (A > o.f1 ? o.f1 : A); break;
			case OP_LERP: r = lerpf(A, B, R[o.c]); break;
			case OP_YGRAD: {  // i0 = from_coordinate, i1/i2 = min/max coordinate, f0 = from_value, f1 = factor
				int y = by < o.i1 ? o.i1 : (by > o.i2 ? o.i2 : by);
				r = o.f0 + (float)(y - o.i0) * o.f1;
				break;
			}
			case OP_SHIFTA:
				r = V ? noiseGet(w, pk, o.i0, bx * 0.25, by * 0.0, bz * 0.25, true, 1.0f) * 4.0f
				      : noiseGet(w, pk, o.i0, bx * 0.25, by * 0.0, bz * 0.25, false, 1.0f) * 4.0f;
				break;
			case OP_SHIFTB:
				r = V ? noiseGet(w, pk, o.i0, bz * 0.25, bx * 0.25, 0.0, true, 4.0f)
				      : noiseGet(w, pk, o.i0, bz * 0.25, bx * 0.25, 0.0, false, 1.0f) * 4.0f;
				break;
			case OP_NOISE:
				if (o.unshifted) r = noiseGet(w, pk, o.i0, bx * o.d0, by * o.d1, bz * o.d0, V, 1.0f);
				else r = noiseGet(w, pk, o.i0, bx * o.d0 + (double)A, by * o.d1 + (double)B, bz * o.d0 + (double)R[o.c], false, 1.0f);
				break;
			case OP_RANGE: r = (A >= o.f0 && A < o.f1) ? B : R[o.c]; break;
			case OP_SPLINE: r = splineEval(pk, o.i0, R); break;
			default: r = 0.0f;
			}
			R[i] = r;
		}
	}
}

HDN void evalOps(const World& w, const PackDev& pk, int bx, int by, int bz, bool vol, bool colHit, int upto, float* out) {
	float pv[MAX_OPS], vv[MAX_OPS];
	evalOpsState(w, pk, bx, by, bz, vol, colHit, upto, pv, vv, false);
	float* src = vol ? vv : pv;
	for (int i = 0; i <= upto; i++) out[i] = src[i];
}

// ---------------------------------------------------------------- biomes (Climate RTree)

HD int64_t quantize(float v) { return (int64_t)(v * 10000.0f); }

HD int64_t pdist(int64_t lo, int64_t hi, int64_t t) {
	int64_t above = t - hi, below = lo - t;
	return above > 0 ? above : (below > 0 ? below : 0);
}

HD int64_t nodeDist(const RNode& n, const int64_t* t) {
	int64_t d = 0;
	for (int i = 0; i < 7; i++) {
		int64_t k = pdist(n.lo[i], n.hi[i], t[i]);
		d += k * k;
	}
	return d;
}

HDN int rtreeSearch(const PackDev& pk, int node, const int64_t* t, int cand) {
	const RNode& n = pk.rnodes[node];
	if (n.nchild == 0) return node;
	int64_t minD = cand < 0 ? INT64_MAX : nodeDist(pk.rnodes[cand], t);
	int best = cand;
	for (int c = 0; c < n.nchild; c++) {
		int ch = n.child0 + c;
		int64_t cd = nodeDist(pk.rnodes[ch], t);
		if (minD > cd) {
			int leaf = rtreeSearch(pk, ch, t, best);
			int64_t ld = leaf == ch ? cd : nodeDist(pk.rnodes[leaf], t);
			if (minD > ld) {
				minD = ld;
				best = leaf;
			}
		}
	}
	return best;
}

// cand = leaf node index to start from (vanilla's RTree.lastResult), -1 = none. Returns the leaf.
HD int leafFromClimate(const PackDev& pk, const float* c6, int cand) {
	int64_t t[7];
	for (int i = 0; i < 6; i++) t[i] = quantize(c6[i]);
	t[6] = 0;
	return rtreeSearch(pk, 0, t, cand);
}
HD int biomeFromClimate(const PackDev& pk, const float* c6) { return pk.rnodes[leafFromClimate(pk, c6, -1)].biome; }

HD int climateUpto(const PackDev& pk) {
	int m = 0;
	for (int i = 0; i < 6; i++) m = pk.climate[i] > m ? pk.climate[i] : m;
	return m;
}

// biome at a quart position. mode 0 = point (uncached), 1 = volume, 2 = point with column-cache hit
HD int biomeAt(const World& w, const PackDev& pk, int qx, int qy, int qz, int mode, float* climateOut = nullptr) {
	float v[MAX_OPS];
	int upto = climateUpto(pk);
	evalOps(w, pk, qx << 2, qy << 2, qz << 2, mode == 1, mode == 2, upto, v);
	float c6[6];
	for (int i = 0; i < 6; i++) c6[i] = v[pk.climate[i]];
	if (climateOut)
		for (int i = 0; i < 6; i++) climateOut[i] = c6[i];
	return biomeFromClimate(pk, c6);
}

// ---------------------------------------------------------------- spawn origin (NoiseSpawnFinder)

HD int64_t spawnFitness(const World& w, const PackDev& pk, int bx, int bz) {
	int qbx = (bx >> 2) << 2, qbz = (bz >> 2) << 2;
	float v[MAX_OPS];
	int upto = 0;
	for (int p = 0; p < pk.nspawn; p++)
		for (int k = 0; k < pk.spawnTerms[p]; k++) upto = pk.spawnOp[p][k] > upto ? pk.spawnOp[p][k] : upto;
	evalOps(w, pk, qbx, 0, qbz, false, false, upto, v);
	int64_t minF = INT64_MAX;
	for (int p = 0; p < pk.nspawn; p++) {
		int64_t f = 0;
		for (int k = 0; k < pk.spawnTerms[p]; k++) {
			int64_t d = pdist(pk.spawnLo[p][k], pk.spawnHi[p][k], quantize(v[pk.spawnOp[p][k]]));
			f += d * d;
		}
		minF = f < minF ? f : minF;
	}
	return minF * (2048LL * 2048LL) + (int64_t)bx * bx + (int64_t)bz * bz;
}

// returns spawn origin block (x,z); the spawn chunk is (x>>4, z>>4)
HD void spawnOrigin(const World& w, const PackDev& pk, int& outX, int& outZ) {
	int bx = 0, bz = 0;
	int64_t best = spawnFitness(w, pk, 0, 0);
	const float steps[2][2] = {{2048.0f, 512.0f}, {512.0f, 32.0f}};
	for (int s = 0; s < 2; s++) {
		float maxR = steps[s][0], inc = steps[s][1];
		float angle = 0.0f, radius = inc;
		int ox = bx, oz = bz;
		while (radius <= maxR) {
			int x = ox + (int)(sin((double)angle) * radius);
			int z = oz + (int)(cos((double)angle) * radius);
			// fitness = climate*2048^2 + x^2+z^2 >= x^2+z^2: a candidate that far out can't win (exact pruning)
			int64_t f = ((int64_t)x * x + (int64_t)z * z >= best) ? INT64_MAX : spawnFitness(w, pk, x, z);
			if (f < best) {
				best = f;
				bx = x;
				bz = z;
			}
			angle += inc / radius;
			if (angle > 6.283185307179586) {
				angle = 0.0f;
				radius += inc;
			}
		}
	}
	outX = bx;
	outZ = bz;
}

// ---------------------------------------------------------------- structure placement (48-bit, LCG only)

HD int floorDiv(int a, int b) {
	int q = a / b;
	return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

struct Spread {
	int spacing, separation, salt, triangular;
	int freqMethod;  // 0 default, 1 legacy_type_1, 2 legacy_type_2, 3 legacy_type_3
	float frequency;
};

HD void spreadCandidate(const Spread& s, uint64_t seed, int regX, int regZ, int& cx, int& cz) {
	JavaRandom r;
	r.setSeed((uint64_t)((int64_t)regX * 341873128712LL + (int64_t)regZ * 132897987541LL) + seed + (uint64_t)(int64_t)s.salt);
	int lim = s.spacing - s.separation;
	int ox, oz;
	if (s.triangular) {
		ox = (r.nextInt(lim) + r.nextInt(lim)) / 2;
		oz = (r.nextInt(lim) + r.nextInt(lim)) / 2;
	} else {
		ox = r.nextInt(lim);
		oz = r.nextInt(lim);
	}
	cx = regX * s.spacing + ox;
	cz = regZ * s.spacing + oz;
}

HD bool frequencyOk(const Spread& s, uint64_t seed, int cx, int cz) {
	if (!(s.frequency < 1.0f)) return true;
	JavaRandom r;
	switch (s.freqMethod) {
	case 0:  // probabilityReducer: setLargeFeatureWithSalt(seed, salt, x, z)  (argument order as in vanilla)
		r.setSeed((uint64_t)((int64_t)s.salt * 341873128712LL + (int64_t)cx * 132897987541LL) + seed + (uint64_t)(int64_t)cz);
		return r.nextFloat() < s.frequency;
	case 1: {  // legacyPillagerOutpostReducer
		int x = cx >> 4, z = cz >> 4;
		r.setSeed((uint64_t)(int64_t)(x ^ (z << 4)) ^ seed);
		r.nextInt();
		return r.nextInt((int)(1.0f / s.frequency)) == 0;
	}
	case 2:
		r.setSeed((uint64_t)((int64_t)cx * 341873128712LL + (int64_t)cz * 132897987541LL) + seed + 10387320ULL);
		return r.nextFloat() < s.frequency;
	default: {  // legacy_type_3: setLargeFeatureSeed + nextDouble
		r.setSeed(seed);
		int64_t xs = r.nextLong(), zs = r.nextLong();
		r.setSeed((uint64_t)((int64_t)cx * xs) ^ (uint64_t)((int64_t)cz * zs) ^ seed);
		return r.nextDouble() < (double)s.frequency;
	}
	}
}

HD bool isSpreadChunk(const Spread& s, uint64_t seed, int cx, int cz) {
	int px, pz;
	spreadCandidate(s, seed, floorDiv(cx, s.spacing), floorDiv(cz, s.spacing), px, pz);
	return px == cx && pz == cz && frequencyOk(s, seed, cx, cz);
}

// weighted structure choice sequence for multi-structure sets (setLargeFeatureSeed)
HD void largeFeatureRandom(JavaRandom& r, uint64_t seed, int cx, int cz) {
	r.setSeed(seed);
	int64_t xs = r.nextLong(), zs = r.nextLong();
	r.setSeed((uint64_t)((int64_t)cx * xs) ^ (uint64_t)((int64_t)cz * zs) ^ seed);
}

}  // namespace sw

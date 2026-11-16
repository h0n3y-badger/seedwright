// Exact terrain height (host only, 26.3+ packs): ChunkGenerator.getBaseHeight(x, z, WORLD_SURFACE_WG), i.e.
// NoiseBasedChunkGenerator.iterateNoiseColumn - the router's final density sampled over the block column through
// 26.3's compiled density samplers (volume + point paths, caches, cell interpolation) and NoiseBasedAquifer.
// Written from vanilla's sources (densityfunction/*, synth/*, Aquifer, NoiseChunk, RandomState).
//
// The pack's "terrain" section (tools/extract.py TerrainCompiler) is the function graph after vanilla's own
// rewrites (inlined references, deduplicated caches, SLICE_UNIFORM_AXES); compile() here picks the same sampler
// variants as the Java compileSampler methods (const add/mul/div forms etc.), since those differ in float rounding.
//
// What is not reproduced: interval-based shortcuts (min/max of non-overlapping ranges, MinSampler/MaxSampler point
// short-circuits) - they never change a value, only which cache cells get filled, and a point query that lands on a
// cached volume reads the volume-path value (equal to ~1 ulp). Legacy packs (1.21.1 - 26.2, double NoiseChunk) have
// no terrain section: Terrain::ok is false there and callers fall back to the estimate.
//
// Use: Terrain T; T.load(packPath);  TerrainWorld tw(T, seed);  int y = tw.height(x, z);   (memoized per column)
#pragma once
#include "sw_core.h"
#include "pack.h"
#include "jigsaw.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace sw {

namespace terr {

inline int floorDivI(int a, int b) {
	int q = a / b;
	return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}
inline int floorModI(int a, int b) { return a - floorDivI(a, b) * b; }
inline float lerpF(float t, float a, float b) { return a + t * (b - a); }
inline double lerpD2(double t, double a, double b) { return a + t * (b - a); }
inline float lerp3F(float a1, float a2, float a3, float x000, float x100, float x010, float x110, float x001, float x101, float x011, float x111) {
	return lerpF(a3, lerpF(a2, lerpF(a1, x000, x100), lerpF(a1, x010, x110)), lerpF(a2, lerpF(a1, x001, x101), lerpF(a1, x011, x111)));
}
inline float clampF(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

struct Vol {  // DensityVolume
	int sx, sy, sz, x0, y0, z0, stx, sty, stz;
	int size() const { return sx * sy * sz; }
	int idx(int x, int y, int z) const { return y + (x + z * sx) * sy; }
	int bx(int x) const { return x0 + x * stx; }
	int by(int y) const { return y0 + y * sty; }
	int bz(int z) const { return z0 + z * stz; }
	int maxX() const { return x0 + (sx - 1) * stx; }
	int maxY() const { return y0 + (sy - 1) * sty; }
	int maxZ() const { return z0 + (sz - 1) * stz; }
	bool operator==(const Vol& o) const { return memcmp(this, &o, sizeof o) == 0; }
	int indexOfBlock(int x, int y, int z) const {
		int rx = x - x0, ry = y - y0, rz = z - z0;
		if (stx == 1 && sty == 1 && stz == 1) {
			if (rx >= 0 && ry >= 0 && rz >= 0 && rx < sx && ry < sy && rz < sz) return idx(rx, ry, rz);
			return -1;
		}
		if (rx >= 0 && ry >= 0 && rz >= 0 && rx < sx * stx && ry < sy * sty && rz < sz * stz && floorModI(rx, stx) == 0 && floorModI(ry, sty) == 0 &&
		    floorModI(rz, stz) == 0)
			return idx(floorDivI(rx, stx), floorDivI(ry, sty), floorDivI(rz, stz));
		return -1;
	}
};
inline Vol vol(int sx, int sy, int sz, int x0, int y0, int z0, int stx = 1, int sty = 1, int stz = 1) { return Vol{sx, sy, sz, x0, y0, z0, stx, sty, stz}; }

using Buf = std::vector<float>;

// ---------------------------------------------------------------- noise (synth.PerlinNoise / SmearedPerlinNoise / NoiseStack)

static const int8_t GRAD[16][3] = {{1, 1, 0}, {-1, 1, 0}, {1, -1, 0}, {-1, -1, 0}, {1, 0, 1}, {-1, 0, 1}, {1, 0, -1}, {-1, 0, -1},
                                   {0, 1, 1}, {0, -1, 1}, {0, 1, -1}, {0, -1, -1}, {1, 1, 0}, {0, -1, 1}, {-1, 1, 0}, {0, -1, -1}};

struct Octave {
	Perlin p;  // offsets + permutation (GradientNoise)
	double freq;
	float amp;
	bool smeared;
	double fudge;  // SmearedPerlinNoise.fudgeYScale

	int perm(int x) const { return p.p[x & 0xFF]; }
	const int8_t* grad(int x) const { return GRAD[perm(x) & 15]; }
	static float dot(const int8_t* g, float x, float y, float z) { return g[0] * x + g[1] * y + g[2] * z; }
	float sampleAndLerp(int x, int y, int z, float rx, float ry, float rz, float ory) const {
		int x0 = perm(x), x1 = perm(x + 1);
		int xy00 = perm(x0 + y), xy01 = perm(x0 + y + 1), xy10 = perm(x1 + y), xy11 = perm(x1 + y + 1);
		float d000 = dot(grad(xy00 + z), rx, ry, rz), d100 = dot(grad(xy10 + z), rx - 1.0f, ry, rz);
		float d010 = dot(grad(xy01 + z), rx, ry - 1.0f, rz), d110 = dot(grad(xy11 + z), rx - 1.0f, ry - 1.0f, rz);
		float d001 = dot(grad(xy00 + z + 1), rx, ry, rz - 1.0f), d101 = dot(grad(xy10 + z + 1), rx - 1.0f, ry, rz - 1.0f);
		float d011 = dot(grad(xy01 + z + 1), rx, ry - 1.0f, rz - 1.0f), d111 = dot(grad(xy11 + z + 1), rx - 1.0f, ry - 1.0f, rz - 1.0f);
		return lerp3F(smoothstep(rx), smoothstep(ory), smoothstep(rz), d000, d100, d010, d110, d001, d101, d011, d111);
	}
	double fudgeY(double originalY, double relY) const {
		double lim = (originalY >= 0.0 && originalY < relY) ? originalY : relY;
		return floor(lim / fudge + (double)1.0E-7F) * fudge;
	}
	float get(double X, double Y, double Z) const {
		double x = wrapCoord(X) + p.ox, y = wrapCoord(Y) + p.oy, z = wrapCoord(Z) + p.oz;
		int fx = (int)floor(x), fy = (int)floor(y), fz = (int)floor(z);
		float rx = (float)(x - fx), rz = (float)(z - fz);
		if (!smeared) {
			float ry = (float)(y - fy);
			return sampleAndLerp(fx, fy, fz, rx, ry, rz, ry);
		}
		double ry = y - fy;
		float fry = (float)(ry - fudgeY(Y, ry));
		return sampleAndLerp(fx, fy, fz, rx, fry, rz, (float)ry);
	}
	// PerlinNoise/SmearedPerlinNoise.addToVolume
	void addToVolume(float* buf, const Vol& v, double xzScale, double yScale, float amplitude) const {
		int index = 0;
		for (int iz = 0; iz < v.sz; iz++) {
			double z = wrapCoord(v.bz(iz) * xzScale) + p.oz;
			int fz = (int)floor(z);
			float rz = (float)(z - fz), az = smoothstep(rz);
			for (int ix = 0; ix < v.sx; ix++) {
				double x = wrapCoord(v.bx(ix) * xzScale) + p.ox;
				int fx = (int)floor(x);
				float rx = (float)(x - fx);
				int x0 = perm(fx), x1 = perm(fx + 1);
				float ax = smoothstep(rx);
				int lastFy = INT32_MIN;
				float d[8] = {0}, g[8] = {0};
				for (int iy = 0; iy < v.sy; iy++) {
					double oy = v.by(iy) * yScale;
					double y = wrapCoord(oy) + p.oy;
					int fy = (int)floor(y);
					double rY = y - fy;
					float ry = (float)rY;
					float ay = smoothstep(ry);
					if (lastFy != fy) {
						int xy00 = perm(x0 + fy), xy01 = perm(x0 + fy + 1), xy10 = perm(x1 + fy), xy11 = perm(x1 + fy + 1);
						const int8_t* G[8] = {grad(xy00 + fz), grad(xy10 + fz), grad(xy01 + fz), grad(xy11 + fz),
						                      grad(xy00 + fz + 1), grad(xy10 + fz + 1), grad(xy01 + fz + 1), grad(xy11 + fz + 1)};
						const float dx[8] = {rx, rx - 1.0f, rx, rx - 1.0f, rx, rx - 1.0f, rx, rx - 1.0f};
						const float dz[8] = {rz, rz, rz, rz, rz - 1.0f, rz - 1.0f, rz - 1.0f, rz - 1.0f};
						for (int k = 0; k < 8; k++) {
							d[k] = G[k][0] * dx[k] + G[k][2] * dz[k];
							g[k] = G[k][1];
						}
						lastFy = fy;
					}
					float t = smeared ? (float)(rY - fudgeY(oy, rY)) : ry;
					float t1 = t - 1.0f;
					buf[index] += amplitude * lerp3F(ax, ay, az, d[0] + g[0] * t, d[1] + g[1] * t, d[2] + g[2] * t1, d[3] + g[3] * t1, d[4] + g[4] * t,
					                                 d[5] + g[5] * t, d[6] + g[6] * t1, d[7] + g[7] * t1);
					index++;
				}
			}
		}
	}
};

struct NoiseStack {
	std::vector<Octave> layers;
	float get(double x, double y, double z) const {
		float v = 0.0f;
		for (auto& L : layers) v += L.amp * L.get(x * L.freq, y * L.freq, z * L.freq);
		return v;
	}
	void addToVolume(float* buf, const Vol& v, double xzScale, double yScale, float amplitude) const {
		for (auto& L : layers) L.addToVolume(buf, v, xzScale * L.freq, yScale * L.freq, amplitude * L.amp);
	}
};

// ---------------------------------------------------------------- pack section

struct TSpline {  // CubicSpline: coord < 0 = constant
	int coord = -1;
	float value = 0;
	std::vector<float> loc, der;
	std::vector<int> child;
};

struct TNodeDesc {
	std::string kind;
	std::vector<int64_t> ints;
	std::vector<int> kids;
	int spline = -1;
};

}  // namespace terr

struct Terrain {
	bool ok = false;
	std::string error;
	int minY = -64, height = 384, seaLevel = 63, aquifers = 0;
	struct NoiseDef {
		std::string name;
		uint64_t hlo, hhi;
		struct L {
			int oct;
			double freq;
			float amp;
		};
		std::vector<L> layers;
	};
	std::vector<NoiseDef> noises;
	std::vector<terr::TNodeDesc> nodes;
	std::vector<terr::TSpline> splines;
	std::vector<int> cacheInput;  // cache id -> node
	std::vector<int> roots;  // final density [, barrier, floodedness, spread, lava, exclusion, surface level]

	bool load(const std::string& path) {
		std::ifstream in(path);
		std::string line;
		while (std::getline(in, line))
			if (line.rfind("terrain ", 0) == 0) break;
		if (!in) {
			error = "pack has no terrain section (exact heights need a 26.3+ pack rebuilt with tools/extract.py)";
			return false;
		}
		{
			std::istringstream ss(line);
			std::string k, fluid;
			int ver;
			ss >> k >> ver >> minY >> height >> seaLevel >> aquifers >> fluid;
		}
		auto header = [&](const char* name) {
			std::getline(in, line);
			std::istringstream ss(line);
			std::string k;
			int n;
			ss >> k >> n;
			if (k != name) throw std::runtime_error(std::string("terrain section: expected ") + name);
			return n;
		};
		int nn = header("terrain_noises");
		for (int i = 0; i < nn; i++) {
			std::getline(in, line);
			std::istringstream ss(line);
			NoiseDef d;
			int nl;
			ss >> d.name >> nl;
			md5pair(d.name, d.hlo, d.hhi);
			for (int l = 0; l < nl; l++) {
				std::string fr, am;
				NoiseDef::L L;
				ss >> L.oct >> fr >> am;
				L.freq = bitsD(fr);
				L.amp = bitsF(am);
				d.layers.push_back(L);
			}
			noises.push_back(d);
		}
		int nd = header("terrain_nodes");
		for (int i = 0; i < nd; i++) {
			std::getline(in, line);
			std::istringstream ss(line);
			terr::TNodeDesc d;
			std::string bar, tok;
			ss >> d.kind >> bar;
			if (d.kind == "spline") {
				ss >> bar;
				d.spline = parseSpline(ss);
			} else {
				while (ss >> tok && tok != "|") d.ints.push_back(tok[0] == '-' ? -(int64_t)std::stoull(tok.substr(1), nullptr, 16) : (int64_t)std::stoull(tok, nullptr, 16));
				while (ss >> tok) d.kids.push_back(std::stoi(tok));
			}
			nodes.push_back(d);
		}
		auto list = [&](const char* name, std::vector<int>& out) {
			std::getline(in, line);
			std::istringstream ss(line);
			std::string k;
			int n;
			ss >> k >> n;
			if (k != name) throw std::runtime_error(std::string("terrain section: expected ") + name);
			out.resize(n);
			for (int& v : out) ss >> v;
		};
		list("terrain_caches", cacheInput);
		list("terrain_roots", roots);
		ok = !roots.empty();
		return ok;
	}

private:
	int parseSpline(std::istringstream& ss) {
		std::string k;
		ss >> k;
		terr::TSpline s;
		if (k == "c") {
			std::string v;
			ss >> v;
			s.value = bitsF(v);
		} else {
			int n;
			ss >> s.coord >> n;
			for (int i = 0; i < n; i++) {
				std::string a, b, paren;
				ss >> a >> b >> paren;
				s.loc.push_back(bitsF(a));
				s.der.push_back(bitsF(b));
				s.child.push_back(parseSpline(ss));
				ss >> paren;
			}
		}
		splines.push_back(s);
		return (int)splines.size() - 1;
	}
};

// Per-seed compiled samplers (RandomState) + a memo of column heights.
class TerrainWorld {
public:
	TerrainWorld(const Terrain& T, int64_t seed) : T(T), seed(seed) { build(); }

	// ChunkGenerator.getBaseHeight(x, z, WORLD_SURFACE_WG, ...): one above the highest non-air block, or minY
	int height(int x, int z) {
		uint64_t key = ((uint64_t)(uint32_t)x << 32) | (uint32_t)z;
		auto it = memo.find(key);
		if (it != memo.end()) return it->second;
		int h = column(x, z);
		memo[key] = h;
		return h;
	}
	long long columns() const { return (long long)memo.size(); }

private:
	using Vol = terr::Vol;
	enum Kind {
		K_CONST, K_ADD, K_SUB, K_MUL, K_DIV, K_MIN, K_MAX, K_CADD, K_CSUB, K_CMUL, K_CDIV, K_CMIN, K_CMAX,
		K_ABS, K_SQUARE, K_CUBE, K_SQRT, K_HALFNEG, K_QUARTNEG, K_RECIP, K_NEGATE, K_SQUEEZE, K_LOG, K_SIGN,
		K_CLAMP, K_LERP, K_RANGE, K_RANGEC, K_ISEL, K_GRADC, K_GRADR, K_GRADM, K_NOISE, K_NOISEXZ, K_NOISEXYZ, K_SHIFTB,
		K_CACHE, K_INTERP, K_SLICEX, K_SLICEY, K_SLICEZ, K_SLICEXZ, K_FINDTOP, K_SPLINE
	};
	struct S {  // compiled sampler
		int kind;
		int a = -1, b = -1, c = -1;  // child samplers
		std::vector<int> kids;
		std::vector<float> th;
		float f0 = 0, f1 = 0, f2 = 0, f3 = 0;
		int i0 = 0, i1 = 0, i2 = 0, i3 = 0;
		double d0 = 0, d1 = 0;
		int noise = -1;
	};
	struct CacheCell {
		bool has = false;
		Vol v{};
		terr::Buf buf;
		int64_t key = 0;
		float value = NAN;
	};

	const Terrain& T;
	int64_t seed;
	std::vector<terr::NoiseStack> noiseObj;  // pack noises, then blended noise stacks
	std::vector<S> smp;
	std::vector<int> nodeSampler;  // pack node -> sampler (compile memo: DensityFunctionCompiler shares compiled children)
	std::vector<int> cacheSampler;  // cache id -> sampler of its input
	int rootFinal = -1, rootBarrier = -1, rootFlood = -1, rootSpread = -1, rootLava = -1, rootExcl = -1, rootSurface = -1;
	uint64_t posLo = 0, posHi = 0, aqLo = 0, aqHi = 0;
	std::vector<CacheCell> cells;
	std::unordered_map<uint64_t, int> memo;

	// ---------------------------------------------------------------- setup

	static float fbits(int64_t v) {
		uint32_t u = (uint32_t)v;
		float f;
		memcpy(&f, &u, 4);
		return f;
	}
	static double dbits(int64_t v) {
		uint64_t u = (uint64_t)v;
		double d;
		memcpy(&d, &u, 8);
		return d;
	}
	int add(S s) {
		smp.push_back(s);
		return (int)smp.size() - 1;
	}
	static S mk(int kind, int a = -1, int b = -1, int c = -1) {
		S s;
		s.kind = kind;
		s.a = a;
		s.b = b;
		s.c = c;
		return s;
	}

	void initPerlin(Perlin& P, Xoro& r) { perlinInit(P, r); }

	void build() {
		Xoro r;
		r.seed64((uint64_t)seed);
		posLo = r.nextLong();
		posHi = r.nextLong();
		for (auto& nd : T.noises) {  // NormalNoise.create(random.fromHashOf(key))
			Xoro nr;
			nr.init(nd.hlo ^ posLo, nd.hhi ^ posHi);
			uint64_t f[4];
			for (int k = 0; k < 4; k++) f[k] = nr.nextLong();
			terr::NoiseStack st;
			for (auto& L : nd.layers) {
				uint64_t hlo, hhi;
				md5pair("octave_" + std::to_string(L.oct), hlo, hhi);
				for (int second = 0; second < 2; second++) {
					terr::Octave o;
					Xoro lr;
					lr.init(hlo ^ f[second * 2], hhi ^ f[second * 2 + 1]);
					perlinInit(o.p, lr);
					o.freq = second ? L.freq * 1.0181268882175227 : L.freq;
					o.amp = L.amp;
					o.smeared = false;
					o.fudge = 0;
					st.layers.push_back(o);
				}
			}
			noiseObj.push_back(std::move(st));
		}
		{  // RandomState.getOrCreateRandomFactory("minecraft:aquifer") = random.fromHashOf(id).forkPositional()
			uint64_t hlo, hhi;
			md5pair("minecraft:aquifer", hlo, hhi);
			Xoro ar;
			ar.init(hlo ^ posLo, hhi ^ posHi);
			aqLo = ar.nextLong();
			aqHi = ar.nextLong();
		}
		nodeSampler.assign(T.nodes.size(), -1);
		cacheSampler.assign(T.cacheInput.size(), -1);
		std::vector<int> roots;
		for (int r0 : T.roots) roots.push_back(compile(r0));
		for (size_t i = 0; i < T.cacheInput.size(); i++)
			if (cacheSampler[i] < 0) cacheSampler[i] = compile(T.cacheInput[i]);
		rootFinal = roots[0];
		if (roots.size() >= 7) {
			rootBarrier = roots[1], rootFlood = roots[2], rootSpread = roots[3], rootLava = roots[4], rootExcl = roots[5], rootSurface = roots[6];
		}
		cells.resize(T.cacheInput.size());
	}

	bool isConst(int node, float* v = nullptr) const {
		const auto& d = T.nodes[node];
		if (d.kind != "const") return false;
		if (v) *v = fbits(d.ints[0]);
		return true;
	}

	int blended(const terr::TNodeDesc& d) {  // BlendedNoise.compileSampler(context.createRandom("minecraft:terrain"))
		double xzScale = dbits(d.ints[0]), yScale = dbits(d.ints[1]), xzFactor = dbits(d.ints[2]), yFactor = dbits(d.ints[3]),
		       smear = dbits(d.ints[4]);
		uint64_t hlo, hhi;
		md5pair("minecraft:terrain", hlo, hhi);
		Xoro r;
		r.init(hlo ^ posLo, hhi ^ posHi);
		double xzMul = 684.412 * xzScale, yMul = 684.412 * yScale;
		double limitSmear = yMul * smear, mainSmear = limitSmear / yFactor;
		auto fbm = [&](int firstOctave, double smearY, double valueFactor) {
			int octaves = -firstOctave + 1;
			double factor = 1.0;
			valueFactor /= std::pow(2.0, octaves) - 1.0;
			terr::NoiseStack st;
			for (int i = octaves - 1; i >= 0; i--) {
				terr::Octave o;
				perlinInit(o.p, r);
				o.smeared = true;
				o.fudge = smearY * factor;
				o.freq = factor;
				o.amp = (float)valueFactor;
				st.layers.push_back(o);
				factor /= 2.0;
				valueFactor *= 2.0;
			}
			noiseObj.push_back(std::move(st));
			return (int)noiseObj.size() - 1;
		};
		int nMin = fbm(-15, limitSmear, (double)0.99998474F), nMax = fbm(-15, limitSmear, (double)0.99998474F), nMain = fbm(-7, mainSmear, 12.75);
		auto noiseS = [&](int n, double xz, double y) {
			S s = mk(K_NOISE);
			s.noise = n;
			s.d0 = xz;
			s.d1 = y;
			return add(s);
		};
		int smin = noiseS(nMin, xzMul, yMul), smax = noiseS(nMax, xzMul, yMul), smain = noiseS(nMain, xzMul / xzFactor, yMul / yFactor);
		S ca = mk(K_CADD, smain);
		ca.f0 = 0.5f;
		S cl = mk(K_CLAMP, add(ca));
		cl.f0 = 0.0f;
		cl.f1 = 1.0f;
		return add(mk(K_LERP, add(cl), smin, smax));
	}

	int compile(int node) {
		if (nodeSampler[node] >= 0) return nodeSampler[node];
		const auto& d = T.nodes[node];
		const std::string& k = d.kind;
		auto kid = [&](int i) { return compile(d.kids[i]); };
		int out;
		if (k == "const") {
			S s = mk(K_CONST);
			s.f0 = fbits(d.ints[0]);
			out = add(s);
		} else if (k == "blend_alpha" || k == "blend_offset" || k == "beardifier") {  // ContextBoundSampler, no blender / beardifier
			S s = mk(K_CONST);
			s.f0 = k == "blend_alpha" ? 1.0f : 0.0f;
			out = add(s);
		} else if (k == "blend_density") {  // no blender: input unchanged
			out = kid(0);
		} else if (k == "add" || k == "sub" || k == "mul" || k == "div") {  // BinaryFunction.compileSampler
			int L = d.kids[0], R = d.kids[1];
			float lc, rc;
			bool lC = isConst(L, &lc), rC = isConst(R, &rc);
			S s;
			if (k == "add") {
				if (lC) (s = mk(K_CADD, compile(R))).f0 = lc;
				else if (rC) (s = mk(K_CADD, compile(L))).f0 = rc;
				else s = mk(K_ADD, compile(L), compile(R));
			} else if (k == "sub") {
				if (lC) (s = mk(K_CSUB, compile(R))).f0 = lc;
				else if (rC) (s = mk(K_CADD, compile(L))).f0 = -rc;
				else s = mk(K_SUB, compile(L), compile(R));
			} else if (k == "mul") {
				if (lC) (s = mk(K_CMUL, compile(R))).f0 = lc;
				else if (rC) (s = mk(K_CMUL, compile(L))).f0 = rc;
				else s = mk(K_MUL, compile(L), compile(R));
			} else {
				if (lC) (s = mk(K_CDIV, compile(R))).f0 = lc;
				else if (rC) (s = mk(K_CMUL, compile(L))).f0 = 1.0f / rc;
				else s = mk(K_DIV, compile(L), compile(R));
			}
			out = add(s);
		} else if (k == "min" || k == "max") {
			int L = d.kids[0], R = d.kids[1];
			float lc, rc;
			S s;
			bool mn = k == "min";
			if (isConst(L, &lc)) (s = mk(mn ? K_CMIN : K_CMAX, compile(R))).f0 = lc;
			else if (isConst(R, &rc)) (s = mk(mn ? K_CMIN : K_CMAX, compile(L))).f0 = rc;
			else s = mk(mn ? K_MIN : K_MAX, compile(L), compile(R));
			out = add(s);
		} else if (k == "abs" || k == "square" || k == "cube" || k == "sqrt" || k == "half_negative" || k == "quarter_negative" ||
		           k == "reciprocal" || k == "negate" || k == "squeeze" || k == "log" || k == "sign") {
			static const char* names[] = {"abs", "square", "cube", "sqrt", "half_negative", "quarter_negative", "reciprocal", "negate", "squeeze", "log", "sign"};
			int kk = K_ABS;
			for (int i = 0; i < 11; i++)
				if (k == names[i]) kk = K_ABS + i;
			out = add(mk(kk, kid(0)));
		} else if (k == "clamp") {
			S s = mk(K_CLAMP, kid(0));
			s.f0 = fbits(d.ints[0]);
			s.f1 = fbits(d.ints[1]);
			out = add(s);
		} else if (k == "lerp") {
			out = add(mk(K_LERP, kid(0), kid(1), kid(2)));
		} else if (k == "range_choice") {
			float a, b;
			S s;
			if (isConst(d.kids[1], &a) && isConst(d.kids[2], &b)) {
				s = mk(K_RANGEC, kid(0));
				s.f2 = a;
				s.f3 = b;
			} else {
				s = mk(K_RANGE, kid(0), kid(1), kid(2));
			}
			s.f0 = fbits(d.ints[0]);
			s.f1 = fbits(d.ints[1]);
			out = add(s);
		} else if (k == "interval_select") {
			S s = mk(K_ISEL, kid(0));
			int n = (int)d.ints[0];
			for (int i = 0; i < n; i++) s.th.push_back(fbits(d.ints[1 + i]));
			for (size_t i = 1; i < d.kids.size(); i++) s.kids.push_back(compile(d.kids[i]));
			out = add(s);
		} else if (k == "gradient") {  // GradientFunction.compileSampler
			int axis = (int)d.ints[0], tiling = (int)d.ints[1], from = (int)d.ints[2], to = (int)d.ints[3];
			float fv = fbits(d.ints[4]), tv = fbits(d.ints[5]);
			int range = to - from;
			float factor = (tv - fv) / (float)range;
			S s = mk(tiling == 0 ? K_GRADC : tiling == 1 ? K_GRADR : K_GRADM);
			s.i0 = axis;
			s.i1 = from;
			s.i2 = tiling == 0 ? std::min(from, to) : range;
			s.i3 = std::max(from, to);
			s.f0 = fv;
			s.f1 = factor;
			out = add(s);
		} else if (k == "noise") {  // NoiseFunction.compileSampler
			int n = (int)d.ints[0];
			double xz = dbits(d.ints[1]), y = dbits(d.ints[2]);
			auto zero = [&](int nd) {
				float v;
				return isConst(nd, &v) && memcmp(&v, "\0\0\0\0", 4) == 0;
			};
			S s;
			if (zero(d.kids[0]) && zero(d.kids[1]) && zero(d.kids[2])) s = mk(K_NOISE);
			else if (zero(d.kids[1])) s = mk(K_NOISEXZ, kid(0), -1, kid(2));
			else s = mk(K_NOISEXYZ, kid(0), kid(1), kid(2));
			s.noise = n;
			s.d0 = xz;
			s.d1 = y;
			out = add(s);
		} else if (k == "shift_a" || k == "shift") {  // ConstMul(NoiseFunction.Sampler(noise, 0.25, 0 | 0.25), 4)
			S s = mk(K_NOISE);
			s.noise = (int)d.ints[0];
			s.d0 = 0.25;
			s.d1 = k == "shift" ? 0.25 : 0.0;
			S m = mk(K_CMUL, add(s));
			m.f0 = 4.0f;
			out = add(m);
		} else if (k == "shift_b") {
			S s = mk(K_SHIFTB);
			s.noise = (int)d.ints[0];
			out = add(s);
		} else if (k == "cache") {
			int id = (int)d.ints[0];
			if (cacheSampler[id] < 0) cacheSampler[id] = compile(T.cacheInput[id]);
			S s = mk(K_CACHE, cacheSampler[id]);
			s.i0 = id;
			out = add(s);
		} else if (k == "interpolated") {
			S s = mk(K_INTERP, kid(0));
			s.i0 = (int)d.ints[0];
			s.i1 = (int)d.ints[1];
			s.f0 = 1.0f / s.i0;
			s.f1 = 1.0f / s.i1;
			out = add(s);
		} else if (k == "slice") {  // SliceFunction.compileSampler (X of Z / Z of X -> XZ sampler)
			int axis = (int)d.ints[0], coord = (int)d.ints[1];
			const auto& in = T.nodes[d.kids[0]];
			if (in.kind == "slice" && ((axis == 0 && in.ints[0] == 2) || (axis == 2 && in.ints[0] == 0))) {
				S s = mk(K_SLICEXZ, compile(in.kids[0]));
				s.i0 = axis == 0 ? coord : (int)in.ints[1];
				s.i1 = axis == 0 ? (int)in.ints[1] : coord;
				out = add(s);
			} else {
				S s = mk(axis == 0 ? K_SLICEX : axis == 1 ? K_SLICEY : K_SLICEZ, kid(0));
				s.i0 = coord;
				out = add(s);
			}
		} else if (k == "find_top") {  // YSampler(FindTopSurface.Sampler, 0)
			S s = mk(K_FINDTOP, kid(0), kid(1));
			s.i0 = (int)d.ints[0];  // lower bound
			s.i1 = (int)d.ints[1];  // cell height
			S y = mk(K_SLICEY, add(s));
			y.i0 = 0;
			out = add(y);
		} else if (k == "blended") {
			out = blended(d);
		} else if (k == "spline") {
			S s = mk(K_SPLINE);
			s.i0 = d.spline;
			collectSplineCoords(d.spline);
			out = add(s);
		} else {
			throw std::runtime_error("terrain: unknown node kind " + k);
		}
		nodeSampler[node] = out;
		return out;
	}
	void collectSplineCoords(int si) {
		const auto& s = T.splines[si];
		if (s.coord < 0) return;
		compile(s.coord);
		for (int c : s.child) collectSplineCoords(c);
	}

	// ---------------------------------------------------------------- point sampling

	float splineValue(int si, int x, int y, int z, std::vector<std::pair<int, float>>& coordMemo) {
		const auto& sn = T.splines[si];
		if (sn.coord < 0) return sn.value;
		int cs = nodeSampler[sn.coord];
		float input = NAN;
		for (auto& p : coordMemo)
			if (p.first == cs) input = p.second;
		if (std::isnan(input)) {
			input = value(cs, x, y, z);
			coordMemo.push_back({cs, input});
		}
		return splineCore(sn, input, [&](int c) { return splineValue(c, x, y, z, coordMemo); });
	}
	template <class F>
	static float splineCore(const terr::TSpline& sn, float input, F child) {
		int n = (int)sn.loc.size();
		int from = 0, len = n;
		while (len > 0) {
			int half = len / 2, mid = from + half;
			if (input < sn.loc[mid]) len = half;
			else {
				from = mid + 1;
				len -= half + 1;
			}
		}
		int start = from - 1, last = n - 1;
		if (start < 0) {
			float v = child(sn.child[0]);
			return sn.der[0] == 0.0f ? v : v + sn.der[0] * (input - sn.loc[0]);
		}
		if (start == last) {
			float v = child(sn.child[last]);
			return sn.der[last] == 0.0f ? v : v + sn.der[last] * (input - sn.loc[last]);
		}
		float x1 = sn.loc[start], x2 = sn.loc[start + 1];
		float t = (input - x1) / (x2 - x1);
		float d1 = sn.der[start], d2 = sn.der[start + 1];
		float y1 = child(sn.child[start]), y2 = child(sn.child[start + 1]);
		float a = d1 * (x2 - x1) - (y2 - y1);
		float b = -d2 * (x2 - x1) + (y2 - y1);
		return terr::lerpF(t, y1, y2) + t * (1.0f - t) * terr::lerpF(t, a, b);
	}

	static float gradCompute(const S& s, int coord) {
		switch (s.kind) {
		case K_GRADC: {
			int c = coord < s.i2 ? s.i2 : (coord > s.i3 ? s.i3 : coord);
			return s.f0 + (float)(c - s.i1) * s.f1;
		}
		case K_GRADR: return s.f0 + (float)terr::floorModI(coord - s.i1, s.i2) * s.f1;
		default: {
			int rel = coord - s.i1, tile = terr::floorDivI(rel, s.i2), local = rel - tile * s.i2;
			return (tile & 1) == 0 ? s.f0 + (float)local * s.f1 : s.f0 + (float)(s.i2 - local) * s.f1;
		}
		}
	}
	static int axisPick(int axis, int x, int y, int z) { return axis == 0 ? x : axis == 1 ? y : z; }

	float value(int si, int x, int y, int z) {
		const S& s = smp[si];
		switch (s.kind) {
		case K_CONST: return s.f0;
		case K_ADD: return value(s.a, x, y, z) + value(s.b, x, y, z);
		case K_SUB: return value(s.a, x, y, z) - value(s.b, x, y, z);
		case K_MUL: {
			float l = value(s.a, x, y, z);
			return l == 0.0f ? 0.0f : l * value(s.b, x, y, z);
		}
		case K_DIV: {
			float l = value(s.a, x, y, z);
			return l == 0.0f ? 0.0f : l / value(s.b, x, y, z);
		}
		case K_MIN: {
			float l = value(s.a, x, y, z), r = value(s.b, x, y, z);
			return l <= r ? l : r;
		}
		case K_MAX: {
			float l = value(s.a, x, y, z), r = value(s.b, x, y, z);
			return l >= r ? l : r;
		}
		case K_CADD: return value(s.a, x, y, z) + s.f0;
		case K_CSUB: return s.f0 - value(s.a, x, y, z);
		case K_CMUL: return value(s.a, x, y, z) * s.f0;
		case K_CDIV: return s.f0 / value(s.a, x, y, z);
		case K_CMIN: {
			float l = value(s.a, x, y, z);
			return l <= s.f0 ? l : s.f0;
		}
		case K_CMAX: {
			float l = value(s.a, x, y, z);
			return l >= s.f0 ? l : s.f0;
		}
		case K_CLAMP: return terr::clampF(value(s.a, x, y, z), s.f0, s.f1);
		case K_LERP: {
			float al = value(s.a, x, y, z);
			if (al == 0.0f) return value(s.b, x, y, z);
			if (al == 1.0f) return value(s.c, x, y, z);
			return terr::lerpF(al, value(s.b, x, y, z), value(s.c, x, y, z));
		}
		case K_RANGEC: {
			float in = value(s.a, x, y, z);
			return in >= s.f0 && in < s.f1 ? s.f2 : s.f3;
		}
		case K_RANGE: {
			float in = value(s.a, x, y, z);
			return in >= s.f0 && in < s.f1 ? value(s.b, x, y, z) : value(s.c, x, y, z);
		}
		case K_ISEL: {
			float in = value(s.a, x, y, z);
			size_t i = 0;
			while (i < s.th.size() && !(in < s.th[i])) i++;
			return value(s.kids[i], x, y, z);
		}
		case K_GRADC: case K_GRADR: case K_GRADM: return gradCompute(s, axisPick(s.i0, x, y, z));
		case K_NOISE: return noiseObj[s.noise].get(x * s.d0, y * s.d1, z * s.d0);
		case K_NOISEXZ: {
			double nx = x * s.d0 + value(s.a, x, y, z);
			double ny = y * s.d1;
			double nz = z * s.d0 + value(s.c, x, y, z);
			return noiseObj[s.noise].get(nx, ny, nz);
		}
		case K_NOISEXYZ: {
			double nx = x * s.d0 + value(s.a, x, y, z);
			double ny = y * s.d1 + value(s.b, x, y, z);
			double nz = z * s.d0 + value(s.c, x, y, z);
			return noiseObj[s.noise].get(nx, ny, nz);
		}
		case K_SHIFTB: return noiseObj[s.noise].get(z * 0.25, x * 0.25, 0.0) * 4.0f;
		case K_CACHE: {
			CacheCell& cell = cells[s.i0];
			int64_t key = blockKey(x, y, z);
			if (cell.key == key && !std::isnan(cell.value)) return cell.value;
			if (cell.has) {
				int i = cell.v.indexOfBlock(x, y, z);
				if (i != -1) return cell.buf[i];
			}
			float v = value(s.a, x, y, z);
			cell.key = key;
			cell.value = v;
			return v;
		}
		case K_INTERP: {
			int cxz = s.i0, cy = s.i1;
			int xi = terr::floorModI(x, cxz), yi = terr::floorModI(y, cy), zi = terr::floorModI(z, cxz);
			if (xi == 0 && yi == 0 && zi == 0) return value(s.a, x, y, z);
			Vol v = terr::vol(2, 2, 2, x - xi, y - yi, z - zi, cxz, cy, cxz);
			terr::Buf b(v.size());
			volume(s.a, b.data(), v);
			return terr::lerp3F((float)xi / cxz, (float)yi / cy, (float)zi / cxz, b[v.idx(0, 0, 0)], b[v.idx(1, 0, 0)], b[v.idx(0, 1, 0)],
			                    b[v.idx(1, 1, 0)], b[v.idx(0, 0, 1)], b[v.idx(1, 0, 1)], b[v.idx(0, 1, 1)], b[v.idx(1, 1, 1)]);
		}
		case K_SLICEX: return value(s.a, s.i0, y, z);
		case K_SLICEY: return value(s.a, x, s.i0, z);
		case K_SLICEZ: return value(s.a, x, y, s.i0);
		case K_SLICEXZ: return value(s.a, s.i0, y, s.i1);
		case K_FINDTOP: return findTop(s, x, z, value(s.b, x, y, z));
		case K_SPLINE: {
			std::vector<std::pair<int, float>> memo;
			return splineValue(s.i0, x, y, z, memo);
		}
		default: return unary(s.kind, value(s.a, x, y, z));
		}
	}

	static float unary(int kind, float v) {
		switch (kind) {
		case K_ABS: return std::fabs(v);
		case K_SQUARE: return v * v;
		case K_CUBE: return v * v * v;
		case K_SQRT: return (float)std::sqrt((double)v);
		case K_HALFNEG: return v > 0.0f ? v : v * 0.5f;
		case K_QUARTNEG: return v > 0.0f ? v : v * 0.25f;
		case K_RECIP: return 1.0f / v;
		case K_NEGATE: return -v;
		case K_SQUEEZE: {
			float c = terr::clampF(v, -1.0f, 1.0f);
			return c / 2.0f - c * c * c / 24.0f;
		}
		case K_LOG: return (float)std::log((double)v);
		case K_SIGN: return v > 0.0f ? 1.0f : v < 0.0f ? -1.0f : v;
		}
		return v;
	}

	float findTop(const S& s, int x, int z, float upper) {
		int topY = (int)floor((double)(upper / (float)s.i1)) * s.i1;
		if (topY <= s.i0) return (float)s.i0;
		for (int py = topY; py >= s.i0; py -= s.i1)
			if (value(s.a, x, py, z) > 0.0f) return (float)py;
		return (float)s.i0;
	}

	static int64_t blockKey(int x, int y, int z) {  // BlockPos.asLong
		return (int64_t)((((uint64_t)x & 0x3FFFFFF) << 38) | (((uint64_t)z & 0x3FFFFFF) << 12) | ((uint64_t)y & 0xFFF));
	}
	static int posX(int64_t L) { return (int)(L >> 38); }
	static int posY(int64_t L) { return (int)((int64_t)((uint64_t)L << 52) >> 52); }
	static int posZ(int64_t L) { return (int)((int64_t)((uint64_t)L << 26) >> 38); }

	// ---------------------------------------------------------------- volume sampling

	void volume(int si, float* out, const Vol& v) {
		const S& s = smp[si];
		int n = v.size();
		switch (s.kind) {
		case K_CONST:
			for (int i = 0; i < n; i++) out[i] = s.f0;
			return;
		case K_ADD: case K_SUB: case K_MUL: case K_DIV: case K_MIN: case K_MAX: {
			volume(s.a, out, v);
			terr::Buf r(n);
			volume(s.b, r.data(), v);
			for (int i = 0; i < n; i++) {
				float l = out[i], rv = r[i];
				switch (s.kind) {
				case K_ADD: out[i] = l + rv; break;
				case K_SUB: out[i] = l + -rv; break;
				case K_MUL: out[i] = l * rv; break;
				case K_DIV: out[i] = l / rv; break;
				case K_MIN: if (rv < l) out[i] = rv; break;
				default: if (rv > l) out[i] = rv;
				}
			}
			return;
		}
		case K_CADD: case K_CSUB: case K_CMUL: case K_CDIV: case K_CMIN: case K_CMAX:
			volume(s.a, out, v);
			for (int i = 0; i < n; i++) {
				float l = out[i];
				switch (s.kind) {
				case K_CADD: out[i] = l + s.f0; break;
				case K_CSUB: out[i] = s.f0 - l; break;
				case K_CMUL: out[i] = l * s.f0; break;
				case K_CDIV: out[i] = s.f0 / l; break;
				case K_CMIN: if (s.f0 < l) out[i] = s.f0; break;
				default: if (s.f0 > l) out[i] = s.f0;
				}
			}
			return;
		case K_CLAMP:
			volume(s.a, out, v);
			for (int i = 0; i < n; i++) out[i] = terr::clampF(out[i], s.f0, s.f1);
			return;
		case K_LERP: {
			volume(s.a, out, v);
			terr::Buf f(n), sc(n);
			volume(s.b, f.data(), v);
			volume(s.c, sc.data(), v);
			for (int i = 0; i < n; i++) {
				float al = out[i];
				out[i] = al == 0.0f ? f[i] : al == 1.0f ? sc[i] : terr::lerpF(al, f[i], sc[i]);
			}
			return;
		}
		case K_RANGEC:
			volume(s.a, out, v);
			for (int i = 0; i < n; i++) out[i] = out[i] >= s.f0 && out[i] < s.f1 ? s.f2 : s.f3;
			return;
		case K_RANGE: {
			volume(s.b, out, v);
			terr::Buf in(n), oor(n);
			volume(s.a, in.data(), v);
			volume(s.c, oor.data(), v);
			for (int i = 0; i < n; i++)
				if (!(in[i] >= s.f0) || !(in[i] < s.f1)) out[i] = oor[i];
			return;
		}
		case K_ISEL: {
			volume(s.a, out, v);
			std::vector<terr::Buf> bufs(s.kids.size(), terr::Buf(n));
			for (size_t k = 0; k < s.kids.size(); k++) volume(s.kids[k], bufs[k].data(), v);
			for (int i = 0; i < n; i++) {
				size_t k = 0;
				while (k < s.th.size() && !(out[i] < s.th[k])) k++;
				out[i] = bufs[k][i];
			}
			return;
		}
		case K_GRADC: case K_GRADR: case K_GRADM:
			for (int z = 0; z < v.sz; z++)
				for (int x = 0; x < v.sx; x++)
					for (int y = 0; y < v.sy; y++) out[v.idx(x, y, z)] = gradCompute(s, axisPick(s.i0, v.bx(x), v.by(y), v.bz(z)));
			return;
		case K_NOISE:
			for (int i = 0; i < n; i++) out[i] = 0.0f;
			noiseObj[s.noise].addToVolume(out, v, s.d0, s.d1, 1.0f);
			return;
		case K_NOISEXZ: case K_NOISEXYZ: {
			volume(s.a, out, v);
			terr::Buf sy(n), sz(n);
			if (s.kind == K_NOISEXYZ) volume(s.b, sy.data(), v);
			volume(s.c, sz.data(), v);
			int index = 0;
			for (int z = 0; z < v.sz; z++) {
				double bz = v.bz(z) * s.d0;
				for (int x = 0; x < v.sx; x++) {
					double bx = v.bx(x) * s.d0;
					for (int y = 0; y < v.sy; y++) {
						double nx = bx + out[index];
						double ny = s.kind == K_NOISEXYZ ? v.by(y) * s.d1 + sy[index] : v.by(y) * s.d1;
						double nz = bz + sz[index];
						out[index] = noiseObj[s.noise].get(nx, ny, nz);
						index++;
					}
				}
			}
			return;
		}
		case K_SHIFTB: {
			Vol tv = terr::vol(v.sz, v.sx, 1, v.z0, v.x0, 0, v.stz, v.stx, 1);
			terr::Buf t(tv.size(), 0.0f);
			noiseObj[s.noise].addToVolume(t.data(), tv, 0.25, 0.25, 4.0f);
			for (int z = 0; z < v.sz; z++)
				for (int x = 0; x < v.sx; x++) {
					float val = t[tv.idx(z, x, 0)];
					int i0 = v.idx(x, 0, z);
					for (int y = 0; y < v.sy; y++) out[i0 + y] = val;
				}
			return;
		}
		case K_CACHE: {  // SamplerContext.sampleVolumeCached
			CacheCell& cell = cells[s.i0];
			if (!cell.has || !(v == cell.v)) {
				cell.has = true;
				cell.v = v;
				cell.buf.assign(n, 0.0f);
				volume(s.a, cell.buf.data(), v);
			}
			memcpy(out, cell.buf.data(), sizeof(float) * n);
			return;
		}
		case K_INTERP: interpVolume(s, out, v); return;
		case K_SLICEX: case K_SLICEY: case K_SLICEZ: case K_SLICEXZ: sliceVolume(s, out, v); return;
		case K_FINDTOP: {
			if (v.sy != 1) throw std::runtime_error("find_top_surface: sizeY != 1");
			volume(s.b, out, v);
			int index = 0;
			for (int z = 0; z < v.sz; z++)
				for (int x = 0; x < v.sx; x++) {
					out[index] = findTop(s, v.bx(x), v.bz(z), out[index]);
					index++;
				}
			return;
		}
		case K_SPLINE: {  // BufferSplineInput: each coordinate sampled over the whole volume when first needed
			std::vector<std::pair<int, terr::Buf>> coords;
			for (int i = 0; i < n; i++) out[i] = splineVolumeAt(s.i0, i, v, coords);
			return;
		}
		default:
			volume(s.a, out, v);
			for (int i = 0; i < n; i++) out[i] = unary(s.kind, out[i]);
			return;
		}
	}

	float splineVolumeAt(int si, int i, const Vol& v, std::vector<std::pair<int, terr::Buf>>& coords) {
		const auto& sn = T.splines[si];
		if (sn.coord < 0) return sn.value;
		int cs = nodeSampler[sn.coord];
		const terr::Buf* b = nullptr;
		for (auto& p : coords)
			if (p.first == cs) b = &p.second;
		if (!b) {
			coords.push_back({cs, terr::Buf(v.size())});
			volume(cs, coords.back().second.data(), v);
			b = &coords.back().second;
		}
		float input = (*b)[i];
		return splineCore(sn, input, [&](int c) { return splineVolumeAt(c, i, v, coords); });
	}

	void sliceVolume(const S& s, float* out, const Vol& v) {
		if (s.kind == K_SLICEX) {
			if (v.sx == 1 && v.x0 == s.i0) return volume(s.a, out, v);
			Vol iv = terr::vol(1, v.sy, v.sz, s.i0, v.y0, v.z0, v.stx, v.sty, v.stz);
			terr::Buf b(iv.size());
			volume(s.a, b.data(), iv);
			int index = 0;
			for (int z = 0; z < v.sz; z++)
				for (int x = 0; x < v.sx; x++)
					for (int y = 0; y < v.sy; y++) out[index++] = b[iv.idx(0, y, z)];
		} else if (s.kind == K_SLICEZ) {
			if (v.sz == 1 && v.z0 == s.i0) return volume(s.a, out, v);
			Vol iv = terr::vol(v.sx, v.sy, 1, v.x0, v.y0, s.i0, v.stx, v.sty, v.stz);
			terr::Buf b(iv.size());
			volume(s.a, b.data(), iv);
			int index = 0;
			for (int z = 0; z < v.sz; z++)
				for (int x = 0; x < v.sx; x++)
					for (int y = 0; y < v.sy; y++) out[index++] = b[iv.idx(x, y, 0)];
		} else if (s.kind == K_SLICEY) {
			if (v.sy == 1 && v.y0 == s.i0) return volume(s.a, out, v);
			Vol iv = terr::vol(v.sx, 1, v.sz, v.x0, s.i0, v.z0, v.stx, v.sty, v.stz);
			terr::Buf b(iv.size());
			volume(s.a, b.data(), iv);
			for (int z = 0; z < v.sz; z++)
				for (int x = 0; x < v.sx; x++) {
					float val = b[iv.idx(x, 0, z)];
					int i0 = v.idx(x, 0, z);
					for (int y = 0; y < v.sy; y++) out[i0 + y] = val;
				}
		} else {
			if (v.sx == 1 && v.sz == 1 && v.x0 == s.i0 && v.z0 == s.i1) return volume(s.a, out, v);
			Vol iv = terr::vol(1, v.sy, 1, s.i0, v.y0, s.i1, v.stx, v.sty, v.stz);
			terr::Buf b(iv.size());
			volume(s.a, b.data(), iv);
			for (int y = 0; y < v.sy; y++) {
				float val = b[iv.idx(0, y, 0)];
				int index = v.idx(0, y, 0);
				for (int z = 0; z < v.sz; z++)
					for (int x = 0; x < v.sx; x++) {
						out[index] = val;
						index += v.sy;
					}
			}
		}
	}

	void interpVolume(const S& s, float* out, const Vol& v) {  // InterpolatedFunction.Sampler.sampleVolume
		int cxz = s.i0, cy = s.i1;
		if ((v.stx == cxz || v.sx == 1) && (v.sty == cy || v.sy == 1) && (v.stz == cxz || v.sz == 1) && terr::floorModI(v.x0, cxz) == 0 &&
		    terr::floorModI(v.y0, cy) == 0 && terr::floorModI(v.z0, cxz) == 0)
			return volume(s.a, out, v);
		if (v.stx == 1 && v.sty == 1 && v.stz == 1) return interpBlockStep(s, out, v);
		Vol bv = terr::vol(v.sx * v.stx, v.sy * v.sty, v.sz * v.stz, v.x0, v.y0, v.z0);
		terr::Buf b(bv.size());
		interpBlockStep(s, b.data(), bv);
		for (int z = 0; z < v.sz; z++)
			for (int x = 0; x < v.sx; x++)
				for (int y = 0; y < v.sy; y++) out[v.idx(x, y, z)] = b[bv.idx(x * v.stx, y * v.sty, z * v.stz)];
	}

	void interpBlockStep(const S& s, float* out, const Vol& v) {
		using terr::floorDivI;
		using terr::floorModI;
		int cxz = s.i0, cy = s.i1;
		int mnx = floorDivI(v.x0, cxz), mny = floorDivI(v.y0, cy), mnz = floorDivI(v.z0, cxz);
		int mxx = floorDivI(v.maxX(), cxz), mxy = floorDivI(v.maxY(), cy), mxz = floorDivI(v.maxZ(), cxz);
		int ccx = mxx - mnx + 1, ccy = mxy - mny + 1, ccz = mxz - mnz + 1;
		Vol cv = terr::vol(floorModI(v.maxX(), cxz) == 0 ? ccx : ccx + 1, floorModI(v.maxY(), cy) == 0 ? ccy : ccy + 1,
		                   floorModI(v.maxZ(), cxz) == 0 ? ccz : ccz + 1, mnx * cxz, mny * cy, mnz * cxz, cxz, cy, cxz);
		terr::Buf cb(cv.size());
		volume(s.a, cb.data(), cv);
		for (int cz = 0; cz < ccz; cz++) {
			int ncz = std::min(cz + 1, cv.sz - 1);
			for (int cx = 0; cx < ccx; cx++) {
				int ncx = std::min(cx + 1, cv.sx - 1);
				float v000 = cb[cv.idx(cx, 0, cz)], v100 = cb[cv.idx(ncx, 0, cz)], v001 = cb[cv.idx(cx, 0, ncz)], v101 = cb[cv.idx(ncx, 0, ncz)];
				for (int cyy = 0; cyy < ccy; cyy++) {
					int ncy = std::min(cyy + 1, cv.sy - 1);
					float v010 = cb[cv.idx(cx, ncy, cz)], v110 = cb[cv.idx(ncx, ncy, cz)], v011 = cb[cv.idx(cx, ncy, ncz)], v111 = cb[cv.idx(ncx, ncy, ncz)];
					// fillCell
					int ox = cv.bx(cx) - v.x0, oy = cv.by(cyy) - v.y0, oz = cv.bz(cz) - v.z0;
					int x0 = std::max(0, -ox), y0 = std::max(0, -oy), z0 = std::max(0, -oz);
					int x1 = std::min(cxz, v.sx - ox) - 1, y1 = std::min(cy, v.sy - oy) - 1, z1 = std::min(cxz, v.sz - oz) - 1;
					for (int z = z0; z <= z1; z++) {
						float az = z * s.f0;
						float v00 = terr::lerpF(az, v000, v001), v01 = terr::lerpF(az, v010, v011);
						float v10 = terr::lerpF(az, v100, v101), v11 = terr::lerpF(az, v110, v111);
						for (int x = x0; x <= x1; x++) {
							float ax = x * s.f0;
							float a0 = terr::lerpF(ax, v00, v10), a1 = terr::lerpF(ax, v01, v11);
							float step = (a1 - a0) * s.f1;
							float val = a0 + step * y0;
							int oi = v.idx(ox + x, oy + y0, oz + z);
							for (int y = y0; y <= y1; y++) {
								out[oi++] = val;
								val += step;
							}
						}
					}
					v000 = v010;
					v100 = v110;
					v001 = v011;
					v101 = v111;
				}
			}
		}
	}

	// ---------------------------------------------------------------- column (iterateNoiseColumn + NoiseBasedAquifer)

	enum Sub { SOLID, AIR, WATER, LAVA };
	struct Fluid {
		int level;
		int type;  // WATER / LAVA / AIR
		int at(int y) const { return y < level ? type : AIR; }
		bool operator==(const Fluid& o) const { return level == o.level && type == o.type; }
	};
	Fluid globalFluid(int y) const { return y < std::min(-54, T.seaLevel) ? Fluid{-54, LAVA} : Fluid{T.seaLevel, WATER}; }

	struct Aq {
		int minGX, minGY, minGZ, sizeGX, sizeGZ, skipAbove;
		std::vector<int64_t> loc;
		std::vector<Fluid> status;
		std::vector<bool> has;
		std::unordered_map<int64_t, int> surface;
	};

	static int gridY(int b) { return terr::floorDivI(b, 12); }
	static int64_t chunkPack(int x, int z) { return ((int64_t)z << 32) | (uint32_t)x; }

	int surfaceLevel(Aq& A, int bx, int bz) {
		int qx = (bx >> 2) << 2, qz = (bz >> 2) << 2;
		auto it = A.surface.find(chunkPack(qx, qz));
		if (it != A.surface.end()) return it->second;
		int v = (int)floor((double)value(rootSurface, qx, 0, qz));
		A.surface[chunkPack(qx, qz)] = v;
		return v;
	}

	Fluid computeFluid(Aq& A, int x, int y, int z) {
		Fluid gf = globalFluid(y);
		static const int OFF[13][2] = {{0, 0}, {-2, -1}, {-1, -1}, {0, -1}, {1, -1}, {-3, 0}, {-2, 0}, {-1, 0}, {1, 0}, {-2, 1}, {-1, 1}, {0, 1}, {1, 1}};
		int lowest = INT32_MAX, top = y + 12, bottom = y - 12;
		bool centerUnder = false;
		for (auto& o : OFF) {
			int sx = x + (o[0] << 4), sz = z + (o[1] << 4);
			int sl = surfaceLevel(A, sx, sz), adj = sl + 8;
			bool start = o[0] == 0 && o[1] == 0;
			if (start && bottom > adj) return gf;
			bool pokes = top > adj;
			if (pokes || start) {
				Fluid gs = globalFluid(adj);
				if (gs.at(adj) != AIR) {
					if (start) centerUnder = true;
					if (pokes) return gs;
				}
			}
			lowest = std::min(lowest, sl);
		}
		// computeSurfaceLevel
		double partial, full;
		if (value(rootExcl, x, y, z) > 0.0f) {
			partial = full = -1.0;
		} else {
			int below = lowest + 8 - y;
			double ff = 0.0;
			if (centerUnder) {  // Mth.clampedMap(below, 0, 64, 1, 0)
				double t = ((double)below - 0.0) / (64.0 - 0.0);
				ff = t < 0.0 ? 1.0 : (t > 1.0 ? 0.0 : terr::lerpD2(t, 1.0, 0.0));
			}
			double fl = (double)value(rootFlood, x, y, z);
			fl = fl < -1.0 ? -1.0 : (fl > 1.0 ? 1.0 : fl);
			double inv = (ff - 1.0) / (0.0 - 1.0);
			double fullT = terr::lerpD2(inv, -0.3, 0.8), partT = terr::lerpD2(inv, -0.8, 0.4);
			partial = fl - partT;
			full = fl - fullT;
		}
		int level;
		if (full > 0.0) level = gf.level;
		else if (partial > 0.0) {
			int fcx = terr::floorDivI(x, 16), fcy = terr::floorDivI(y, 40), fcz = terr::floorDivI(z, 16);
			int mid = fcy * 40 + 20;
			double spread = (double)(value(rootSpread, fcx, fcy, fcz) * 10.0f);
			int q = (int)floor(spread / 3.0) * 3;
			level = std::min(lowest, mid + q);
		} else {
			level = -2032 * 16;  // DimensionType.WAY_BELOW_MIN_Y
		}
		int type = gf.type;
		if (level <= -10 && level != -2032 * 16 && gf.type != LAVA) {
			int tx = terr::floorDivI(x, 64), ty = terr::floorDivI(y, 40), tz = terr::floorDivI(z, 64);
			if (std::fabs((double)value(rootLava, tx, ty, tz)) > 0.3) type = LAVA;
		}
		return Fluid{level, type};
	}

	static uint64_t mthSeed(int x, int y, int z) {  // Mth.getSeed (x * 3129871 is an int multiply)
		int64_t s = (int64_t)(int32_t)((uint32_t)x * 3129871u) ^ ((int64_t)z * 116129781LL) ^ (int64_t)y;
		uint64_t u = (uint64_t)s;
		u = u * u * 42317861ULL + u * 11ULL;
		return (uint64_t)((int64_t)u >> 16);
	}

	double pressure(int x, int y, int z, double& barrier, const Fluid& s1, const Fluid& s2) {
		int t1 = s1.at(y), t2 = s2.at(y);
		if ((t1 == LAVA && t2 == WATER) || (t1 == WATER && t2 == LAVA)) return 2.0;
		int diff = std::abs(s1.level - s2.level);
		if (diff == 0) return 0.0;
		double avg = 0.5 * (s1.level + s2.level);
		double above = y + 0.5 - avg;
		double base = diff / 2.0;
		double toMid = base - std::fabs(above);
		double g;
		if (above > 0.0) {
			double c = 0.0 + toMid;
			g = c > 0.0 ? c / 1.5 : c / 2.5;
		} else {
			double c = 3.0 + toMid;
			g = c > 0.0 ? c / 3.0 : c / 10.0;
		}
		double nv;
		if (!(g < -2.0) && !(g > 2.0)) {
			if (std::isnan(barrier)) barrier = (double)value(rootBarrier, x, y, z);
			nv = barrier;
		} else {
			nv = 0.0;
		}
		return 2.0 * (nv + g);
	}

	int substance(Aq& A, int x, int y, int z, double density) {
		if (density > 0.0) return SOLID;
		Fluid gf = globalFluid(y);
		if (y > A.skipAbove) return gf.at(y);
		if (gf.at(y) == LAVA) return LAVA;
		int xa = (x - 5) >> 4, ya = gridY(y + 1), za = (z - 5) >> 4;
		int d1 = INT32_MAX, d2 = INT32_MAX, d3 = INT32_MAX, i1 = 0, i2 = 0, i3 = 0;
		for (int x1 = 0; x1 <= 1; x1++)
			for (int y1 = -1; y1 <= 1; y1++)
				for (int z1 = 0; z1 <= 1; z1++) {
					int gx = xa + x1, gy = ya + y1, gz = za + z1;
					int index = ((gy - A.minGY) * A.sizeGZ + (gz - A.minGZ)) * A.sizeGX + (gx - A.minGX);
					int64_t L = A.loc[index];
					if (L == INT64_MAX) {
						Xoro r;
						r.init(mthSeed(gx, gy, gz) ^ aqLo, aqHi);
						int ox = r.nextInt(10), oy = r.nextInt(9), oz = r.nextInt(10);
						L = blockKey((gx << 4) + ox, gy * 12 + oy, (gz << 4) + oz);
						A.loc[index] = L;
					}
					int lx = posX(L), ly = posY(L), lz = posZ(L);
					int dx = lx - x, dy = ly - y, dz = lz - z;
					int nd = dx * dx + dy * dy + dz * dz;
					if (d1 >= nd) {
						i3 = i2; i2 = i1; i1 = index;
						d3 = d2; d2 = d1; d1 = nd;
					} else if (d2 >= nd) {
						i3 = i2; i2 = index;
						d3 = d2; d2 = nd;
					} else if (d3 >= nd) {
						i3 = index;
						d3 = nd;
					}
				}
		auto status = [&](int index) -> const Fluid& {
			if (!A.has[index]) {
				int64_t L = A.loc[index];
				A.status[index] = computeFluid(A, posX(L), posY(L), posZ(L));
				A.has[index] = true;
			}
			return A.status[index];
		};
		auto sim = [](int a, int b) { return 1.0 - (b - a) / 25.0; };
		Fluid s1 = status(i1);
		double s12 = sim(d1, d2);
		int fluid = s1.at(y);
		if (s12 <= 0.0) return fluid;
		if (fluid == WATER && globalFluid(y - 1).at(y - 1) == LAVA) return fluid;
		double barrier = NAN;
		Fluid s2 = status(i2);
		if (density + s12 * pressure(x, y, z, barrier, s1, s2) > 0.0) return SOLID;
		Fluid s3 = status(i3);
		double s13 = sim(d1, d3);
		if (s13 > 0.0 && density + s12 * s13 * pressure(x, y, z, barrier, s1, s3) > 0.0) return SOLID;
		double s23 = sim(d2, d3);
		if (s23 > 0.0 && density + s12 * s23 * pressure(x, y, z, barrier, s2, s3) > 0.0) return SOLID;
		return fluid;
	}

	int column(int x, int z) {
		for (auto& c : cells) c = CacheCell();
		Vol col = terr::vol(1, T.height, 1, x, T.minY, z);
		Aq A;
		bool aq = T.aquifers && rootSurface >= 0;
		if (aq) {  // NoiseBasedAquifer constructor
			A.minGX = ((x - 5) >> 4);
			int maxGX = ((col.maxX() - 5) >> 4) + 1;
			A.sizeGX = maxGX - A.minGX + 1;
			A.minGY = gridY(T.minY + 1) - 1;
			int maxGY = gridY(col.maxY() + 1) + 1;
			int sizeGY = maxGY - A.minGY + 1;
			A.minGZ = ((z - 5) >> 4);
			int maxGZ = ((col.maxZ() - 5) >> 4) + 1;
			A.sizeGZ = maxGZ - A.minGZ + 1;
			int total = A.sizeGX * sizeGY * A.sizeGZ;
			A.loc.assign(total, INT64_MAX);
			A.status.assign(total, Fluid{0, AIR});
			A.has.assign(total, false);
			// maxSurfaceLevel(fromGridX(minGX, 0), fromGridZ(minGZ, 0), fromGridX(maxGX, 9), fromGridZ(maxGZ, 9))
			int bx0 = A.minGX << 4, bz0 = A.minGZ << 4, bx1 = (maxGX << 4) + 9, bz1 = (maxGZ << 4) + 9;
			int qx0 = bx0 >> 2, qx1 = bx1 >> 2, qz0 = bz0 >> 2, qz1 = bz1 >> 2;
			Vol sv = terr::vol(qx1 - qx0 + 1, 1, qz1 - qz0 + 1, qx0 << 2, 0, qz0 << 2, 4, 1, 4);
			terr::Buf sb(sv.size());
			volume(rootSurface, sb.data(), sv);
			int maxY = INT32_MIN;
			for (int zz = 0; zz < sv.sz; zz++)
				for (int xx = 0; xx < sv.sx; xx++) {
					int sl = (int)floor((double)sb[sv.idx(xx, 0, zz)]);
					A.surface[chunkPack(sv.bx(xx), sv.bz(zz))] = sl;
					maxY = std::max(maxY, sl);
				}
			int adj = maxY + 8;
			int skipGY = gridY(adj + 12) + 1;
			A.skipAbove = skipGY * 12 + 11 - 1;
		}
		terr::Buf d(col.size());
		volume(rootFinal, d.data(), col);
		for (int yi = col.sy - 1; yi >= 0; yi--) {
			int by = col.by(yi);
			double dens = d[col.idx(0, yi, 0)];
			int sub = aq ? substance(A, x, by, z, dens) : (dens > 0.0 ? SOLID : globalFluid(by).at(by));
			if (sub != AIR) return by + 1;
		}
		return T.minY;
	}
};

// HeightGrid for jigsaw.h that asks tw for every column (host only)
inline int terrainHeightFn(void* ctx, int x, int z) { return static_cast<TerrainWorld*>(ctx)->height(x, z); }
inline HeightGrid terrainGrid(TerrainWorld& tw) {
	HeightGrid g{nullptr, 0, 0, 0, 0};
	g.fn = terrainHeightFn;
	g.ctx = &tw;
	return g;
}

}  // namespace sw

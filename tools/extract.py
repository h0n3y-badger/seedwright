#!/usr/bin/env python3
"""Build a seedwright version pack from a vanilla Minecraft client/server jar.

Everything comes from the jar's own data pack (worldgen JSON + tags), plus a biome parameter
list dumped by the oracle mod (biome_parameters.txt, see oracle/). Output is a plain-text pack the
C++ engine loads at runtime: packs/<version>.pack

usage: extract.py <jar> <version> <biome_parameters.txt> <out.pack>
"""
import json, struct, sys, zipfile, math

def f32(x):
    return struct.unpack('<f', struct.pack('<f', float(x)))[0]

def f32bits(x):
    return struct.unpack('<I', struct.pack('<f', float(x)))[0]

def dbits(x):
    return struct.unpack('<Q', struct.pack('<d', float(x)))[0]

class Jar:
    def __init__(self, path):
        self.z = zipfile.ZipFile(path)
        self.names = set(self.z.namelist())

    def json(self, kind, ident):
        ns, _, path = ident.partition(':') if ':' in ident else ('minecraft', '', ident)
        return json.loads(self.z.read(f'data/{ns}/{kind}/{path}.json'))

    def list(self, kind):
        pre = f'data/minecraft/{kind}/'
        return sorted(n[len(pre):-5] for n in self.names if n.startswith(pre) and n.endswith('.json'))

    def tag(self, kind, ident, seen=None):
        """Resolve a tag (e.g. worldgen/biome) to a sorted list of ids."""
        seen = seen or set()
        if ident in seen:
            return []
        seen.add(ident)
        out = []
        for v in self.json(f'tags/{kind}', ident)['values']:
            if isinstance(v, dict):
                v = v['id']
            if v.startswith('#'):
                out += self.tag(kind, v[1:], seen)
            else:
                out.append(v if ':' in v else 'minecraft:' + v)
        return out

    def holder_set(self, kind, v):
        if isinstance(v, str):
            return self.tag(kind, v[1:]) if v.startswith('#') else [v]
        return list(v)

# ---------------- noises (NormalNoise constructor, both the 1.21.x and 26.x formats) ----------------

def noise_layers(p):
    """-> list of (octave_index, frequency(double), value_factor(float)) in NoiseStack layer order,
    each octave giving two layers (first, second*1.0181268882175227). Mirrors NormalNoise exactly."""
    if 'firstOctave' in p:  # 1.21.x legacy format: firstOctave + amplitudes, legacy normalization
        return None
    base_octave = p['base_octave']
    count = p.get('octave_count', 1)
    base_amp = p.get('base_amplitude', 1.0)
    norm = p.get('normalize', True)
    mods = p.get('amplitude_modifiers', [])
    mode = 'legacy' if norm == 'legacy' else ('on' if norm else 'off')

    def modifier(i):
        return 1.0 if not mods else mods[i]

    def build(base_amp, normalize):
        freq = math.pow(2.0, base_octave)
        amp = base_amp
        if normalize:
            amp *= math.pow(0.5, -(count - 1)) / (math.pow(0.5, -count) - 1.0)
        octs = []
        for i in range(count):
            m = modifier(i)
            if m != 0.0:
                octs.append((base_octave + i, freq, amp * m))
            freq *= 2.0
            amp *= 0.5
        return octs

    def norm_factor(target_amp, octs):
        var = 0.0
        for o in octs:
            d = 0.2702247831245211 * abs(o[2])
            var += d * d
        dev = math.sqrt(var)
        if dev == 0.0:
            return 0.0
        return (target_amp * 0.3333333333333333) / (dev * math.sqrt(2.0))

    def parity_norm(base_amp):
        lo, hi = 10**9, -10**9
        for i in range(count):
            if modifier(i) != 0.0:
                lo, hi = min(lo, i), max(hi, i)
        return base_amp * 0.5 * 0.3333333333333333 / (0.1 * (1.0 + 1.0 / (hi - lo + 1)))

    octs = build(base_amp, mode != 'off')
    target = sum(abs(o[2]) for o in octs)
    nf = norm_factor(target, octs)
    if mode == 'legacy' and nf != 0.0:
        nf = parity_norm(base_amp)
    return [(o[0], o[1], f32(nf * o[2])) for o in octs]

# ---------------- density functions -> flat point program ----------------

class DfCompiler:
    """Compiles named density functions into a flat op list (post-order, shared nodes deduplicated by
    registry name). Each op: opcode, args (other op indices), float params, noise index."""

    def __init__(self, jar):
        self.jar = jar
        self.ops = []
        self.named = {}
        self.noises = []
        self.noise_index = {}
        self.splines = []

    def noise(self, name):
        if name not in self.noise_index:
            self.noise_index[name] = len(self.noises)
            self.noises.append(name)
        return self.noise_index[name]

    def emit(self, op, args=(), fp=(), dp=(), ip=(), cached=False):
        self.ops.append((op, list(args), [f32bits(x) for x in fp], [dbits(x) for x in dp], list(ip), cached))
        return len(self.ops) - 1

    def ref(self, name):
        if name not in self.named:
            self.named[name] = self.node(self.jar.json('worldgen/density_function', name))
        return self.named[name]

    def node(self, n, cached=False):
        if isinstance(n, (int, float)):
            return self.emit('const', fp=[n])
        if isinstance(n, str):
            return self.ref(n)
        t = n['type'].removeprefix('minecraft:')
        if t in ('cache', 'cache_2d', 'flat_cache', 'cache_once', 'cache_all_in_cell', 'interpolated'):
            i = self.node(n['input'] if 'input' in n else n['argument'])
            if t in ('cache', 'cache_2d', 'flat_cache'):
                op = self.ops[i]
                self.ops[i] = op[:5] + (True,)
            return i
        if t == 'constant':
            return self.emit('const', fp=[n['argument'] if 'argument' in n else n['value']])
        if t in ('add', 'mul', 'min', 'max'):
            a = self.node(n.get('left', n.get('argument1')))
            b = self.node(n.get('right', n.get('argument2')))
            return self.emit(t, [a, b])
        if t in ('abs', 'square', 'cube', 'half_negative', 'quarter_negative', 'squeeze', 'invert'):
            return self.emit(t, [self.node(n.get('input', n.get('argument')))])
        if t == 'clamp':
            return self.emit('clamp', [self.node(n['input'])], fp=[n['min'], n['max']])
        if t == 'blend_alpha':
            return self.emit('const', fp=[1.0])
        if t == 'blend_offset':
            return self.emit('const', fp=[0.0])
        if t == 'blend_density':
            return self.node(n['argument'] if 'argument' in n else n['input'])
        if t == 'lerp':
            return self.emit('lerp', [self.node(n['alpha']), self.node(n['first']), self.node(n['second'])])
        if t in ('y_clamped_gradient',):
            return self.emit('ygrad', fp=[n['from_value'], n['to_value']], ip=[n['from_y'], n['to_y']])
        if t == 'gradient':
            assert n['axis'] == 'y' and n.get('tiling', 'clamp_to_edge') == 'clamp_to_edge', n
            return self.emit('ygrad', fp=[n['from_value'], n['to_value']], ip=[n['from_coordinate'], n['to_coordinate']])
        if t in ('shift_a', 'shift_b', 'shift'):
            return self.emit(t, ip=[self.noise(n['argument'] if 'argument' in n else n['noise'])])
        if t in ('noise', 'shifted_noise'):
            xz = n['xz_scale']
            y = n['y_scale']
            sx = self.node(n.get('shift_x', 0.0))
            sy = self.node(n.get('shift_y', 0.0))
            sz = self.node(n.get('shift_z', 0.0))
            return self.emit('noise', [sx, sy, sz], dp=[xz, y], ip=[self.noise(n['noise'])])
        if t == 'range_choice':
            return self.emit('range_choice', [self.node(n['input']), self.node(n['when_in_range']), self.node(n['when_out_of_range'])],
                             fp=[n['min_inclusive'], n['max_exclusive']])
        if t == 'spline':
            return self.emit('spline', ip=[self.spline(n['spline'])])
        raise ValueError(f'unsupported density function type {t}')

    def spline(self, s):
        """Splines go into their own table: ('c', value) | ('m', coord_op, [(loc, deriv, child)])."""
        if isinstance(s, (int, float)):
            self.splines.append(('c', f32bits(s)))
            return len(self.splines) - 1
        coord = self.node(s['coordinate'])
        pts = [(f32bits(p['location']), f32bits(p.get('derivative', 0.0)), self.spline(p['value'])) for p in s['points']]
        self.splines.append(('m', coord, pts))
        return len(self.splines) - 1

# ---------------- structures ----------------

def structure_entry(jar, name):
    s = jar.json('worldgen/structure', name)
    t = s['type'].removeprefix('minecraft:')
    e = {'name': name, 'type': t, 'biomes': jar.holder_set('worldgen/biome', s['biomes'])}
    if t == 'jigsaw':
        e['project'] = s.get('project_start_to_heightmap', '')
    sh = s.get('start_height')
    y = 'surface'
    if isinstance(sh, dict):
        if 'absolute' in sh:
            y = str(sh['absolute']) if not e.get('project') else 'surface'
        elif sh.get('type', '').endswith('uniform'):
            lo = sh['min_inclusive'].get('absolute', 0); hi = sh['max_inclusive'].get('absolute', 0)
            y = str((lo + hi) // 2)
    if t == 'mineshaft':
        y = '50'
    e['y'] = y
    return e

def main():
    jar_path, version, params_path, out_path = sys.argv[1:5]
    jar = Jar(jar_path)
    out = []
    w = out.append
    w(f'seedwright-pack 1 {version}')

    # biomes, in the order of the dumped parameter list (that order is part of the RTree build)
    params = [l.split() for l in open(params_path) if l.strip() and not l.startswith('#')]
    biome_ids = []
    for p in params:
        if p[0] not in biome_ids:
            biome_ids.append(p[0])
    bidx = {b: i for i, b in enumerate(biome_ids)}
    w(f'biomes {len(biome_ids)}')
    for b in biome_ids:
        w(b)
    w(f'params {len(params)}')
    for p in params:
        w(' '.join([str(bidx[p[0]])] + p[1:]))  # 13 quantized longs (min,max)x6 + offset

    # density functions for the climate sampler
    dfc = DfCompiler(jar)
    ns = jar.json('worldgen/noise_settings', 'overworld')
    router = ns['noise_router']
    roots = [dfc.node(router[k]) for k in ('temperature', 'vegetation', 'continents', 'erosion', 'depth', 'ridges')]
    w('climate ' + ' '.join(map(str, roots)))
    w(f'world {ns["noise"]["min_y"]} {ns["noise"]["height"]} {ns["sea_level"]}')
    spawn = ns.get('spawn_target', [])
    spawn_lines = [f'spawn_target {len(spawn)}']
    for pt in spawn:
        # each point: map of climate parameter -> [min,max] (26.x: keyed by density function; 1.21: fixed fields)
        if all(k in pt for k in ('temperature', 'humidity')):
            keys = ['temperature', 'humidity', 'continentalness', 'erosion', 'depth', 'weirdness']
            vals = []
            for k in keys:
                v = pt[k]
                vals += v if isinstance(v, list) else [v, v]
            spawn_lines.append('fixed ' + ' '.join(str(int(f32(f32(x) * f32(10000.0)))) for x in vals) + f' {int(f32(f32(pt.get("offset", 0.0)) * f32(10000.0)))}')
        else:
            items = []
            for k, v in pt.items():
                v = v if isinstance(v, list) else [v, v]
                items.append(f'{dfc.node(k)} {int(f32(f32(v[0]) * f32(10000.0)))} {int(f32(f32(v[1]) * f32(10000.0)))}')
            spawn_lines.append(f'df {len(items)} ' + ' '.join(items))
    w(f'noises {len(dfc.noises)}')
    for name in dfc.noises:
        layers = noise_layers(jar.json('worldgen/noise', name))
        w(f'{name} {len(layers)} ' + ' '.join(f'{o} {dbits(fr):x} {f32bits(a):x}' for o, fr, a in layers))
    w(f'ops {len(dfc.ops)}')
    for op, args, fp, dp, ip, cached in dfc.ops:
        w(f'{op} {int(cached)} {len(args)} {" ".join(map(str, args))} | {len(fp)} {" ".join(f"{x:x}" for x in fp)} | '
          f'{len(dp)} {" ".join(f"{x:x}" for x in dp)} | {len(ip)} {" ".join(map(str, ip))}')
    w(f'splines {len(dfc.splines)}')
    for s in dfc.splines:
        if s[0] == 'c':
            w(f'c {s[1]:x}')
        else:
            w(f'm {s[1]} {len(s[2])} ' + ' '.join(f'{l:x} {d:x} {c}' for l, d, c in s[2]))
    for l in spawn_lines:
        w(l)
    # structure sets
    sets = []
    for name in jar.list('worldgen/structure_set'):
        s = jar.json('worldgen/structure_set', name)
        sets.append((name, s))
    structs = {}
    w(f'structure_sets {len(sets)}')
    for name, s in sets:
        p = s['placement']
        t = p['type'].removeprefix('minecraft:')
        excl = p.get('exclusion_zone')
        ent = s['structures']
        for e in ent:
            if e['structure'] not in structs:
                structs[e['structure']] = structure_entry(jar, e['structure'])
        sl = ' '.join(f'{e["structure"]} {e["weight"]}' for e in ent)
        freq = f32(p.get('frequency', 1.0))
        frm = p.get('frequency_reduction_method', 'default')
        ex = f'{excl["other_set"]} {excl["chunk_count"]}' if excl else '- 0'
        if t == 'random_spread':
            w(f'{name} random_spread {p["salt"]} {f32bits(freq):x} {frm} {ex} {p["spacing"]} {p["separation"]} '
              f'{p.get("spread_type", "linear")} {len(ent)} {sl}')
        elif t == 'concentric_rings':
            pb = jar.holder_set('worldgen/biome', p['preferred_biomes'])
            w(f'{name} concentric_rings {p.get("salt", 0)} {f32bits(freq):x} {frm} {ex} {p["distance"]} {p["spread"]} {p["count"]} '
              f'{len(ent)} {sl} {len(pb)} {" ".join(pb)}')
        else:
            w(f'{name} {t} 0 0 default - 0 {len(ent)} {sl}')
    w(f'structures {len(structs)}')
    for name, e in structs.items():
        w(f'{name} {e["type"]} {e.get("project", "-") or "-"} {e["y"]} {len(e["biomes"])} {" ".join(e["biomes"])}')
    open(out_path, 'w').write('\n'.join(out) + '\n')
    print(f'{out_path}: {len(biome_ids)} biomes, {len(params)} params, {len(dfc.ops)} ops, {len(dfc.noises)} noises, '
          f'{len(sets)} structure sets')

if __name__ == '__main__':
    main()

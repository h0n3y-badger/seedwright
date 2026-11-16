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
            return self.emit('const', fp=[n], dp=[n])
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
            v = n['argument'] if 'argument' in n else n['value']
            return self.emit('const', fp=[v], dp=[v])
        if t in ('add', 'mul', 'min', 'max'):
            a = self.node(n.get('left', n.get('argument1')))
            b = self.node(n.get('right', n.get('argument2')))
            return self.emit(t, [a, b])
        if t in ('abs', 'square', 'cube', 'half_negative', 'quarter_negative', 'squeeze', 'invert'):
            return self.emit(t, [self.node(n.get('input', n.get('argument')))])
        if t == 'clamp':
            return self.emit('clamp', [self.node(n['input'])], fp=[n['min'], n['max']], dp=[n['min'], n['max']])
        if t == 'blend_alpha':
            return self.emit('const', fp=[1.0], dp=[1.0])
        if t == 'blend_offset':
            return self.emit('const', fp=[0.0], dp=[0.0])
        if t == 'blend_density':
            return self.node(n['argument'] if 'argument' in n else n['input'])
        if t == 'lerp':
            return self.emit('lerp', [self.node(n['alpha']), self.node(n['first']), self.node(n['second'])])
        if t in ('y_clamped_gradient',):
            return self.emit('ygrad', fp=[n['from_value'], n['to_value']], dp=[n['from_value'], n['to_value']], ip=[n['from_y'], n['to_y']])
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
                             fp=[n['min_inclusive'], n['max_exclusive']], dp=[n['min_inclusive'], n['max_exclusive']])
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

# ---------------- jigsaw structures (template pools + structure templates) ----------------

def read_nbt(data):
    """Minimal NBT reader (gzip'd or raw) -> python dicts/lists."""
    import gzip
    if data[:2] == b'\x1f\x8b':
        data = gzip.decompress(data)
    pos = 0

    def rd(fmt):
        nonlocal pos
        v = struct.unpack_from('>' + fmt, data, pos)
        pos += struct.calcsize('>' + fmt)
        return v[0]

    def rstr():
        nonlocal pos
        n = rd('H')
        s = data[pos:pos + n].decode('utf-8', 'replace')
        pos += n
        return s

    def payload(t):
        nonlocal pos
        if t in (1, 2, 3, 4, 5, 6):
            return rd(' bhiqfd'[t])
        if t == 7:
            n = rd('i')
            pos += n
            return None
        if t == 8:
            return rstr()
        if t == 9:
            et = rd('b')
            n = rd('i')
            return [payload(et) for _ in range(n)]
        if t == 10:
            d = {}
            while True:
                tt = rd('b')
                if tt == 0:
                    return d
                k = rstr()
                d[k] = payload(tt)
        if t in (11, 12):
            n = rd('i')
            return [rd('i' if t == 11 else 'q') for _ in range(n)]
        raise ValueError(f'nbt tag {t}')

    t = rd('b')
    rstr()
    return payload(t)

DIRS = ['down', 'up', 'north', 'south', 'west', 'east']  # net.minecraft.core.Direction order

def ns(ident):
    return ident if ':' in ident else 'minecraft:' + ident

def jigsaw_section(jar, version, jigsaw_structs):
    """Template pools, element table and per-template jigsaw blocks for every jigsaw structure.

    Template jigsaw order = StructureTemplate's palette block order for block-entity blocks: sorted by
    (y, x, z). FeaturePoolElement's synthetic jigsaw: name "minecraft:bottom" up to 26.2, null (matches any
    target) in 26.3."""
    names, name_idx = [], {}

    def nid(s):
        if s not in name_idx:
            name_idx[s] = len(names)
            names.append(s)
        return name_idx[s]

    pool_json = {}
    for p in jar.list('worldgen/template_pool'):
        pool_json['minecraft:' + p] = jar.json('worldgen/template_pool', 'minecraft:' + p)
    pools, pool_idx = [], {}
    templates, tpl_idx = [], {}
    elements = []
    queue = []

    def pool_ref(name):
        name = ns(name)
        if name not in pool_json:
            return -1
        if name not in pool_idx:
            pool_idx[name] = len(pools)
            pools.append({'name': name})
            queue.append(name)
        return pool_idx[name]

    def template(loc):
        loc = ns(loc)
        if loc in tpl_idx:
            return tpl_idx[loc]
        nsp, _, path = loc.partition(':')
        fn = f'data/{nsp}/structure/{path}.nbt'
        # a missing template is an empty one in vanilla (StructureTemplateManager.getOrCreate): size 0, no blocks
        d = read_nbt(jar.z.read(fn)) if fn in jar.names else {'size': [0, 0, 0], 'blocks': [], 'palette': []}
        pals = d['palettes'] if 'palettes' in d else [d['palette']]
        jig = []
        for b in d['blocks']:
            s = pals[0][b['state']]
            if s.get('Name', s.get('id')) != 'minecraft:jigsaw':
                continue
            props = s.get('Properties', s.get('properties', {}))
            front, top = props['orientation'].split('_')
            nbt = b.get('nbt', {})
            # StructureTemplate.getDefaultJointType: horizontal front -> aligned, vertical -> rollable
            joint = nbt.get('joint', 'aligned' if front not in ('up', 'down') else 'rollable')
            jig.append((b['pos'][1], b['pos'][0], b['pos'][2], DIRS.index(front), DIRS.index(top), joint,
                        ns(nbt.get('name', 'minecraft:empty')), ns(nbt.get('target', 'minecraft:empty')),
                        ns(nbt.get('pool', 'minecraft:empty')), nbt.get('placement_priority', 0), nbt.get('selection_priority', 0)))
        if len(pals) > 1:  # palette choice depends on position; fine as long as the jigsaws agree
            for pal in pals[1:]:
                other = [b['pos'] for b in d['blocks'] if pal[b['state']].get('Name', pal[b['state']].get('id')) == 'minecraft:jigsaw']
                assert len(other) == len(jig), f'{loc}: palettes disagree on jigsaws'
        jig.sort(key=lambda j: (j[0], j[1], j[2]))  # buildInfoList comparator: y, x, z
        for j in jig:
            pool_ref(j[8])
        tpl_idx[loc] = len(templates)
        templates.append({'name': loc, 'size': d['size'], 'jig': jig})
        return tpl_idx[loc]

    def element(e):
        t = e['element_type'].removeprefix('minecraft:')
        proj = e.get('projection', '-')
        idx = len(elements)
        elements.append(None)
        if t in ('single_pool_element', 'legacy_single_pool_element'):
            loc = e['location'] if isinstance(e['location'], str) else None
            elements[idx] = ('single' if t.startswith('single') else 'legacy', proj, template(loc), [])
        elif t == 'feature_pool_element':
            f = e['feature'] if isinstance(e['feature'], str) else 'inline'
            elements[idx] = ('feature', proj, ns(f), [])
        elif t == 'list_pool_element':
            subs = []
            for s in e['elements']:
                s = dict(s)
                s['projection'] = proj  # ListPoolElement.setProjectionOnEachElement
                subs.append(element(s))
            elements[idx] = ('list', proj, '-', subs)
        elif t == 'empty_pool_element':
            elements[idx] = ('empty', '-', '-', [])
        else:
            raise ValueError(f'pool element type {t}')
        return idx

    structs = []
    for sname in jigsaw_structs:
        s = jar.json('worldgen/structure', sname)
        aliases = 1 if s.get('pool_aliases') else 0
        md = s['max_distance_from_center']
        h, v = (md, md) if isinstance(md, int) else (md['horizontal'], md.get('vertical', 4064))
        sh = s['start_height']
        if 'absolute' in sh:
            hk = ('const', sh['absolute'], sh['absolute'])
        elif sh.get('type', '').endswith('uniform'):
            hk = ('uniform', sh['min_inclusive']['absolute'], sh['max_inclusive']['absolute'])
        else:
            hk = ('unsupported', 0, 0)
        pad = s.get('dimension_padding', 0)
        pb, pt = (pad, pad) if isinstance(pad, int) else (pad.get('bottom', 0), pad.get('top', 0))
        sj = s.get('start_jigsaw_name')
        structs.append((sname, pool_ref(s['start_pool']), nid(ns(sj)) if sj else -1, s['size'], h, v, hk,
                        s.get('project_start_to_heightmap', '-'), 1 if s.get('use_expansion_hack') else 0, pb, pt, aliases))
    while queue:
        pname = queue.pop(0)
        p = pool_json[pname]
        rec = pools[pool_idx[pname]]
        rec['entries'] = [(element(e['element']), e['weight']) for e in p['elements']]
        rec['fallback'] = p['fallback']
        pool_ref(p['fallback'])
    # resolve after the closure is complete
    for p in pools:
        p['fb'] = pool_idx.get(ns(p['fallback']), -1)

    def ybox(ei):  # element bounding box y span at Rotation.NONE (StructureTemplatePool.getMaxSize)
        k, _, a, subs = elements[ei]
        if k in ('single', 'legacy'):
            sy = templates[a]['size'][1]
            return abs(sy - 1) + 1
        if k == 'feature':
            return 1
        if k == 'list':
            lo, hi = 0, 0
            for s in subs:
                if elements[s][0] == 'empty':
                    continue
                sk, _, sa, _ = elements[s]
                top = (templates[sa]['size'][1] - 1) if sk in ('single', 'legacy') else 0
                lo, hi = min(lo, top), max(hi, top)
            return hi - lo + 1
        return 0

    out = [f'jigsaw 1 {"*" if version.startswith("26.3") else "minecraft:bottom"}']
    # pool names first (nid order is independent)
    for p in pools:
        size = sum(wt for _, wt in p['entries'])
        p['size'] = size
    for t in templates:
        for j in t['jig']:
            nid(j[6]); nid(j[7])
    out.append(f'jigsaw_names {len(names)}')
    out += names
    out.append(f'jigsaw_templates {len(templates)}')
    for t in templates:
        js = ' '.join(f'{x} {y} {z} {f} {tp} {0 if jt == "rollable" else 1} {name_idx[n]} {name_idx[tg]} {pool_idx.get(pl, -1)} {pp} {sp}'
                      for (y, x, z, f, tp, jt, n, tg, pl, pp, sp) in t['jig'])
        out.append(f'{t["name"]} {t["size"][0]} {t["size"][1]} {t["size"][2]} {len(t["jig"])} {js}'.rstrip())
    out.append(f'jigsaw_elements {len(elements)}')
    for k, proj, a, subs in elements:
        out.append(f'{k} {proj} {a} {len(subs)} {" ".join(map(str, subs))}'.rstrip())
    out.append(f'jigsaw_pools {len(pools)}')
    for p in pools:
        sizes = [ybox(e) for e, _ in p['entries'] if elements[e][0] != 'empty']
        out.append(f'{p["name"]} {p["fb"]} {max(sizes) if sizes else 0} {len(p["entries"])} ' +
                   ' '.join(f'{e} {wt}' for e, wt in p['entries']))
    out.append(f'jigsaw_structures {len(structs)}')
    for (sname, sp, sj, size, h, v, hk, proj, exp, pb, pt, al) in structs:
        out.append(f'{sname} {sp} {sj} {size} {h} {v} {hk[0]} {hk[1]} {hk[2]} {proj} {exp} {pb} {pt} {al}')
    return out

def main():
    jar_path, version, params_path, out_path = sys.argv[1:5]
    jar = Jar(jar_path)
    out = []
    w = out.append
    legacy = 'firstOctave' in jar.json('worldgen/noise', 'temperature')
    # NoiseSpawnFinder fitness: the quartic distance penalty only exists in 1.21.1 (verified against the oracle)
    old_spawn = version in ('1.21', '1.21.1')
    w(f'seedwright-pack 1 {version} {"legacy" if legacy else "float"} {"oldspawn" if old_spawn else "spawn"}')

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
        p = jar.json('worldgen/noise', name)
        if 'firstOctave' in p:  # legacy (<= 26.2): PerlinNoise octaves, raw amplitudes, computed in the engine
            w(f'{name} legacy {p["firstOctave"]} {len(p["amplitudes"])} ' + ' '.join(f'{dbits(a):x}' for a in p['amplitudes']))
        else:
            layers = noise_layers(p)
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
    # appended last so older loaders (which stop after "structures") still read the pack
    out.extend(jigsaw_section(jar, version, [n for n, e in structs.items() if e['type'] == 'jigsaw']))
    open(out_path, 'w').write('\n'.join(out) + '\n')
    print(f'{out_path}: {len(biome_ids)} biomes, {len(params)} params, {len(dfc.ops)} ops, {len(dfc.noises)} noises, '
          f'{len(sets)} structure sets')

if __name__ == '__main__':
    main()

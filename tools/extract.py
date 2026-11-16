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

# ---------------- terrain (26.3+ float density samplers: final density + aquifer) ----------------

class TerrainCompiler:
    """Mirrors 26.3's DensityFunctionCompiler for the functions getBaseHeight needs: references inlined, caches
    deduplicated by their (raw) input, then DfRewriteRule.SLICE_UNIFORM_AXES. Emits a flat node table (hash-consed,
    children first) that engine/terrain.h compiles into the same sampler variants vanilla picks."""

    def __init__(self, jar):
        self.jar = jar
        self.caches = {}  # canonical raw input -> (id, rewritten input)
        self.cache_inputs = []
        self.noises, self.noise_index = [], {}
        self.nodes, self.node_index = [], {}

    def noise(self, name):
        name = name if ':' in name else 'minecraft:' + name
        if name not in self.noise_index:
            self.noise_index[name] = len(self.noises)
            self.noises.append(name)
        return self.noise_index[name]

    # ---- pass 1: inline references, prepare caches (DensityFunctionCompiler.optimizerRule, first rule)
    def p1(self, n):
        if isinstance(n, str):
            return self.p1(self.jar.json('worldgen/density_function', n))
        if isinstance(n, (int, float)):
            return ('const', f32bits(n))
        t = n['type'].removeprefix('minecraft:')
        g = n.get
        if t == 'cache':
            key = json.dumps(n['input'], sort_keys=True)
            if key not in self.caches:
                cid = len(self.cache_inputs)
                self.cache_inputs.append(None)
                inp = self.slice(self.p1(n['input']), 7)
                self.cache_inputs[cid] = inp
                self.caches[key] = cid
            return ('pcache', self.caches[key])
        if t == 'constant':
            return ('const', f32bits(g('argument', g('value'))))
        if t in ('add', 'sub', 'mul', 'div', 'min', 'max'):
            return (t, self.p1(n['left']), self.p1(n['right']))
        if t in ('abs', 'square', 'cube', 'sqrt', 'half_negative', 'quarter_negative', 'reciprocal', 'negate', 'squeeze', 'log', 'sign',
                 'blend_density'):
            return (t, self.p1(n['input']))
        if t == 'clamp':
            return ('clamp', f32bits(n['min']), f32bits(n['max']), self.p1(n['input']))
        if t == 'lerp':
            return ('lerp', self.p1(n['alpha']), self.p1(n['first']), self.p1(n['second']))
        if t == 'range_choice':
            return ('range_choice', f32bits(n['min_inclusive']), f32bits(n['max_exclusive']), self.p1(n['input']),
                    self.p1(n['when_in_range']), self.p1(n['when_out_of_range']))
        if t == 'interval_select':
            return ('interval_select', tuple(f32bits(x) for x in n['thresholds']), self.p1(n['input'])) + tuple(self.p1(f) for f in n['functions'])
        if t == 'gradient':
            return ('gradient', 'xyz'.index(n['axis']), n.get('tiling', 'clamp_to_edge'), n['from_coordinate'], n['to_coordinate'],
                    f32bits(n['from_value']), f32bits(n['to_value']))
        if t == 'noise':
            return ('noise', self.noise(n['noise']), dbits(n.get('xz_scale', 1.0)), dbits(n.get('y_scale', 1.0)),
                    self.p1(n.get('shift_x', 0.0)), self.p1(n.get('shift_y', 0.0)), self.p1(n.get('shift_z', 0.0)))
        if t in ('shift_a', 'shift_b', 'shift'):
            return (t, self.noise(n.get('noise', n.get('argument'))))
        if t == 'spline':
            return ('spline', self.p1spline(n['spline']))
        if t == 'interpolated':
            return ('interpolated', n['cell_size_xz'], n['cell_size_y'], self.p1(n['input']))
        if t == 'find_top_surface':
            return ('find_top', n['lower_bound'], n['cell_height'], self.p1(n['density']), self.p1(n['upper_bound']))
        if t == 'old_blended_noise':
            return ('blended', dbits(n['xz_scale']), dbits(n['y_scale']), dbits(n['xz_factor']), dbits(n['y_factor']),
                    dbits(n['smear_scale_multiplier']))
        if t in ('blend_alpha', 'blend_offset', 'beardifier'):
            return (t,)
        raise ValueError(f'terrain: unsupported density function type {t}')

    def p1spline(self, s):
        if isinstance(s, (int, float)):
            return ('c', f32bits(s))
        return ('m', self.p1(s['coordinate']), tuple((f32bits(p['location']), f32bits(p.get('derivative', 0.0)), self.p1spline(p['value']))
                                                     for p in s['points']))

    # ---- DensityFunction.domainAxes (x=1, y=2, z=4)
    def axes(self, t):
        k = t[0]
        if k == 'const':
            return 0
        if k in ('blend_alpha', 'blend_offset', 'shift_a', 'shift_b'):
            return 5
        if k in ('beardifier', 'shift', 'blended'):
            return 7
        if k == 'gradient':
            return 1 << t[1] if t[1] != 2 else 4
        if k == 'noise':
            a = 7
            if struct.unpack('<d', struct.pack('<Q', t[3]))[0] == 0.0:
                a &= ~2
            if struct.unpack('<d', struct.pack('<Q', t[2]))[0] == 0.0:
                a &= ~5
            return a | self.axes(t[4]) | self.axes(t[5]) | self.axes(t[6])
        if k == 'pcache':
            return self.axes(self.cache_inputs[t[1]])
        if k == 'slice':
            return self.axes(t[3]) & ~(4 if t[1] == 2 else 1 << t[1])
        if k == 'find_top':
            return (self.axes(t[3]) | self.axes(t[4])) & ~2
        if k == 'spline':
            return self.spline_axes(t[1])
        a = 0
        for c in self.children(t):
            a |= self.axes(c)
        return a

    def spline_axes(self, s):
        if s[0] == 'c':
            return 0
        a = self.axes(s[1])
        for _, _, v in s[2]:
            a |= self.spline_axes(v)
        return a

    @staticmethod
    def child_slots(t):
        k = t[0]
        if k in ('clamp',):
            return [3]
        if k == 'range_choice':
            return [3, 4, 5]
        if k == 'interval_select':
            return list(range(2, len(t)))
        if k == 'noise':
            return [4, 5, 6]
        if k == 'interpolated':
            return [3]
        if k == 'find_top':
            return [3, 4]
        if k == 'slice':
            return [3]
        if k in ('const', 'gradient', 'shift_a', 'shift_b', 'shift', 'blended', 'blend_alpha', 'blend_offset', 'beardifier', 'pcache', 'spline'):
            return []
        return list(range(1, len(t)))

    def children(self, t):
        return [t[i] for i in self.child_slots(t)]

    def rewrite_children(self, t, f):
        if t[0] == 'spline':
            return ('spline', self.spline_map(t[1], f))
        slots = self.child_slots(t)
        if not slots:
            return t
        l = list(t)
        for i in slots:
            l[i] = f(l[i])
        return tuple(l)

    def spline_map(self, s, f):
        if s[0] == 'c':
            return s
        return ('m', f(s[1]), tuple((a, b, self.spline_map(v, f)) for a, b, v in s[2]))

    # ---- pass 2: DfRewriteRule.SliceUniformAxes
    def slice(self, t, parent):
        if t[0] in ('const', 'gradient'):
            return t
        a = self.axes(t)
        if a == parent:
            return self.rewrite_children(t, lambda c: self.slice(c, parent))
        nt = self.rewrite_children(t, lambda c: self.slice(c, a))
        removed = parent & ~a
        existing, u = 0, nt
        while u[0] == 'slice':
            existing |= 4 if u[1] == 2 else 1 << u[1]
            u = u[3]
        removed &= ~existing
        if removed & 1:
            nt = ('slice', 0, 0, nt)
        if removed & 4:
            nt = ('slice', 2, 0, nt)
        if removed & 2:
            nt = ('slice', 1, 0, nt)
        return nt

    def root(self, n):
        return self.emit(self.slice(self.p1(n), 7))

    # ---- emission: "kind | ints | children" per node, children first (deduplicated)
    def emit(self, t):
        if t in self.node_index:
            return self.node_index[t]
        k = t[0]
        if k == 'spline':
            line = 'spline | | ' + self.emit_spline(t[1])
        elif k == 'pcache':
            line = f'cache | {t[1]:x} |'
        else:
            kids = [self.emit(c) for c in self.children(t)]
            slots = set(self.child_slots(t))
            ints = []
            for i, v in enumerate(t[1:], 1):
                if i in slots:
                    continue
                if isinstance(v, tuple):
                    ints += [len(v)] + list(v)
                elif isinstance(v, str):
                    ints.append(['clamp_to_edge', 'repeat', 'mirrored_repeat'].index(v))
                else:
                    ints.append(v)
            line = f'{k} | {" ".join(f"{x:x}" if x >= 0 else "-" + format(-x, "x") for x in ints)} | {" ".join(map(str, kids))}'
        self.nodes.append(line)
        self.node_index[t] = len(self.nodes) - 1
        return self.node_index[t]

    def emit_spline(self, s):
        if s[0] == 'c':
            return f'c {s[1]:x}'
        coord = self.emit(s[1])
        return f'm {coord} {len(s[2])} ' + ' '.join(f'{a:x} {b:x} ( {self.emit_spline(v)} )' for a, b, v in s[2])


def terrain_section(jar, ns):
    """Final density + aquifer for ChunkGenerator.getBaseHeight (26.3 float samplers only)."""
    tc = TerrainCompiler(jar)
    router = ns['noise_router']
    aq = ns.get('aquifers')
    roots = [tc.root(router['final_density'])]
    if aq:
        roots += [tc.root(aq[k]) for k in ('barrier', 'fluid_level_floodedness', 'fluid_level_spread', 'lava', 'exclusion', 'surface_level')]
    caches = [tc.emit(c) for c in tc.cache_inputs]
    out = [f'terrain 1 {ns["noise"]["min_y"]} {ns["noise"]["height"]} {ns["sea_level"]} {1 if aq else 0} {ns["default_fluid"]}']
    out.append(f'terrain_noises {len(tc.noises)}')
    for name in tc.noises:
        layers = noise_layers(jar.json('worldgen/noise', name))
        out.append(f'{name} {len(layers)} ' + ' '.join(f'{o} {dbits(fr):x} {f32bits(a):x}' for o, fr, a in layers))
    out.append(f'terrain_nodes {len(tc.nodes)}')
    out += tc.nodes
    out.append('terrain_caches ' + ' '.join(map(str, [len(caches)] + caches)))
    out.append('terrain_roots ' + ' '.join(map(str, [len(roots)] + roots)))
    return out


# ---------------- chest loot ----------------

# item modifiers that never touch the loot random (anything else marks the table unsupported)
LOOT_NO_RANDOM = {'set_potion', 'set_name', 'set_lore', 'set_components', 'set_custom_data', 'set_item', 'copy_components',
                  'set_custom_model_data', 'set_banner_pattern', 'set_fireworks', 'set_firework_explosion', 'set_book_cover',
                  'set_written_book_pages', 'set_writable_book_pages', 'toggle_tooltips', 'set_contents', 'reference'}

def loot_section(jar, templates, structs_step):
    """Loot tables the jigsaw templates' containers (and buried treasure) use, compiled to: pools (rolls range),
    entries (item, weight, set_count range). Only what the item totals need: modifiers that use the loot random
    other than set_count (enchantments, damage, instruments...) make a table 'unsupported' (its chests count as
    unknown), as do conditions. Also every structure's (generation step, index in the registry within that step)
    for WorldgenRandom.setFeatureSeed (registry order = identifier order, checked with the oracle)."""
    items, item_idx = [], {}
    def item(n):
        n = ns(n)
        if n not in item_idx:
            item_idx[n] = len(items)
            items.append(n)
        return item_idx[n]
    tables, table_idx = [], {}

    def num(v):  # (min, max) of an int provider; None = not a plain constant/uniform
        if isinstance(v, (int, float)):
            return (math.floor(v), math.floor(v))
        if isinstance(v, dict):
            t = v.get('type', 'minecraft:uniform').removeprefix('minecraft:')
            if t == 'constant':
                return num(v['value'])
            if t == 'uniform' and isinstance(v.get('min'), (int, float)) and isinstance(v.get('max'), (int, float)):
                return (math.floor(v['min']), math.floor(v['max']))
        return None

    def mods(o):  # item modifiers of an entry / pool / table, 26.3 'modifier' (one or a list) or older 'functions'
        m = o.get('modifier', o.get('functions', []))
        return m if isinstance(m, list) else [m]

    def table(name):
        name = ns(name)
        if name in table_idx:
            return table_idx[name]
        nsp, _, path = name.partition(':')
        fn = f'data/{nsp}/loot_table/{path}.json'
        ok = fn in jar.names
        pools = []
        d = json.loads(jar.z.read(fn)) if ok else {}
        for m in mods(d):
            ok &= isinstance(m, dict) and m.get('type', m.get('function', '')).removeprefix('minecraft:') in LOOT_NO_RANDOM and 'conditions' not in m and 'condition' not in m
        for p in d.get('pools', []):
            rolls = num(p.get('rolls', 1))
            bonus = p.get('bonus_rolls', 0)
            ok &= rolls is not None and isinstance(bonus, (int, float))  # a constant bonus * luck 0 draws nothing
            ok &= 'conditions' not in p and 'condition' not in p
            for m in mods(p):
                ok &= isinstance(m, dict) and m.get('type', m.get('function', '')).removeprefix('minecraft:') in LOOT_NO_RANDOM
            ents = []
            for e in p.get('entries', []):
                t = e.get('type', '').removeprefix('minecraft:')
                ok &= 'conditions' not in e and 'condition' not in e and t in ('item', 'empty')
                cnt, add = (1, 1), 0
                for m in mods(e):
                    mt = m.get('type', m.get('function', '')).removeprefix('minecraft:') if isinstance(m, dict) else '?'
                    if mt == 'set_count' and 'conditions' not in m and 'condition' not in m and cnt == (1, 1) and num(m['count']) is not None:
                        cnt, add = num(m['count']), 1 if m.get('add') else 0
                    else:
                        ok &= mt in LOOT_NO_RANDOM
                w = e.get('weight', 1)
                q = e.get('quality', 0)
                ents.append((item(e['name']) if t == 'item' else -1, w, q, cnt[0], cnt[1], add))
            pools.append((rolls or (0, 0), ents))
        table_idx[name] = len(tables)
        tables.append((name, 1 if ok else 0, pools))
        return table_idx[name]

    tl = []
    for i, t in enumerate(templates):
        if t['cont']:
            tl.append(f'{i} {len(t["cont"])} ' + ' '.join(f'{x} {y} {z} {table(lt) if lt != "-" else -1}' for (y, x, z, lt) in t['cont']))
    bt = table('minecraft:chests/buried_treasure')
    out = [f'loot 1 {bt}']
    out.append(f'loot_structures {len(structs_step)}')
    out += [f'{n} {st} {ix}' for n, st, ix in structs_step]
    out.append(f'loot_templates {len(tl)}')
    out += tl
    out.append(f'loot_tables {len(tables)}')
    for name, ok, pools in tables:
        out.append(f'{name} {ok} {len(pools)} ' + ' '.join(f'{r[0]} {r[1]} {len(ents)} ' + ' '.join(' '.join(map(str, e)) for e in ents)
                                                        for r, ents in pools))
    out.append(f'loot_items {len(items)}')
    out += items
    return out

def structure_steps(jar):
    """(id, generation step ordinal, index within the step) in registry (= identifier) order."""
    steps = ['raw_generation', 'lakes', 'local_modifications', 'underground_structures', 'surface_structures', 'strongholds',
             'underground_ores', 'underground_decoration', 'fluid_springs', 'vegetal_decoration', 'top_layer_modification']
    per, out = {}, []
    for n in sorted(jar.list('worldgen/structure')):
        st = steps.index(jar.json('worldgen/structure', 'minecraft:' + n)['step'])
        out.append(('minecraft:' + n, st, per.get(st, 0)))
        per[st] = per.get(st, 0) + 1
    return out

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
# block entities implementing RandomizableContainer (+ every *shulker_box)
CONTAINER_BLOCKS = {'minecraft:chest', 'minecraft:trapped_chest', 'minecraft:barrel', 'minecraft:dispenser', 'minecraft:dropper',
                    'minecraft:hopper', 'minecraft:decorated_pot', 'minecraft:crafter'}

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
        # loot containers: block-entity blocks (nbt present) that are RandomizableContainers; each draws
        # random.nextLong() for its LootTableSeed when placed, in palette order = (y, x, z) among nbt blocks
        cont = []
        for b in d['blocks']:
            s = pals[0][b['state']]
            bid = s.get('Name', s.get('id'))
            if 'nbt' in b and (bid in CONTAINER_BLOCKS or bid.endswith('shulker_box')):
                cont.append((b['pos'][1], b['pos'][0], b['pos'][2], ns(b['nbt']['LootTable']) if 'LootTable' in b['nbt'] else '-'))
        cont.sort(key=lambda c: (c[0], c[1], c[2]))
        templates.append({'name': loc, 'size': d['size'], 'jig': jig, 'cont': cont})
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

    # FeaturePoolElement's jigsaw name: "minecraft:bottom" up to 26.2, null since 26.3 (read from the class when
    # the jar is unobfuscated; obfuscated jars are all <= 1.21.11)
    cls = 'net/minecraft/world/level/levelgen/structure/pools/FeaturePoolElement.class'
    fname = 'minecraft:bottom' if cls not in jar.names or b'bottom' in jar.z.read(cls) else '*'
    out = [f'jigsaw 1 {fname}']
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
    jigsaw_section.templates = templates  # for loot_section
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
    if not legacy:  # exact terrain heights (26.3 float samplers); after jigsaw so older loaders ignore it
        out.extend(terrain_section(jar, ns))
    # chest loot (container lists per jigsaw template, loot tables); found by scanning for its header line
    out.extend(loot_section(jar, jigsaw_section.templates, structure_steps(jar)))
    open(out_path, 'w').write('\n'.join(out) + '\n')
    print(f'{out_path}: {len(biome_ids)} biomes, {len(params)} params, {len(dfc.ops)} ops, {len(dfc.noises)} noises, '
          f'{len(sets)} structure sets')

if __name__ == '__main__':
    main()

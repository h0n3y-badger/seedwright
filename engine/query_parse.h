// Query language (host only). The GUI's blocks are a direct picture of this grammar.
//
//   query   := or
//   or      := and ( 'or' and )*
//   and     := unary ( ('and' | ',') unary )*
//   unary   := 'not' unary | atom
//   atom    := '(' or ')'
//            | 'all' '{' or '}'
//            | 'any' '{' alt ( ',' alt )* '}'          alt := and-without-commas
//            | 'biome' NAME 'within' N [ 'at' 'y' '=' (INT | 'any') ]
//            | 'spawn' 'within' N                        (world spawn within N of 0,0)
//            | [COUNT] 'stronghold'[s] 'within' N [ '{' or '}' ]
//            | [COUNT] STRUCTURE 'within' N [ '{' or '}' ]
//            | [COUNT] ['building'] CATEGORY             (only inside a village's braces)
//            | 'loot' [COUNT] ITEM                       (inside a village's or buried treasure's braces: that many
//                                                          of the item in its chests, e.g. 'loot 2 diamond')
//
// STRUCTURE is a structure set ("villages", "village"), an exact structure ("village_plains"), or a
// pattern with '*' ("village_*", "*ruin*"). Everything inside '{ }' is measured from that structure.
// Top-level distances are measured from the world spawn (or 0,0 with --origin).
#pragma once
#include "pack.h"
#include "query.h"
#include <cstring>
#include <algorithm>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sw {

// Vanilla's file names differ per village biome (weaponsmith / weapon_smith, butcher_shop / butchers_shop...):
// buildings are matched by a canonical name, and job names work too (blacksmith, librarian, cleric...).
inline std::string buildingKey(std::string n) {
	static const char* alias[][2] = {
		{"blacksmith", "weaponsmith"}, {"weapon_smith", "weaponsmith"}, {"weaponsmiths", "weaponsmith"},
		{"tool_smith", "toolsmith"}, {"armorer_house", "armorer"}, {"butcher_shop", "butcher"}, {"butchers_shop", "butcher"},
		{"masons_house", "mason"}, {"mason_house", "mason"}, {"shepherds_house", "shepherd"}, {"shepherd_house", "shepherd"},
		{"fisher_cottage", "fisher"}, {"fletcher_house", "fletcher"}, {"cartographer_house", "cartographer"},
		{"librarian", "library"}, {"cleric", "temple"}, {"church", "temple"}, {"leatherworker", "tannery"},
	};
	for (auto& c : n) c = (char)tolower((unsigned char)c);
	for (auto& a : alias)
		if (n == a[0]) return a[1];
	return n;
}

struct QueryNames {
	const Pack* P;
	// building category names per pack structure index (empty: no village layouts in this pack/build)
	std::vector<std::vector<std::pair<std::string, int>>> buildings;
	// chest loot (loot.h): item names, per pack structure which items its chests can hold and whether it needs a
	// jigsaw layout (villages) or not (buried treasure); empty: no loot section in this pack
	std::vector<std::string> lootItems;
	std::vector<std::vector<char>> lootGives;
	std::vector<char> lootJig;
};

class QueryParser {
public:
	QueryParser(const QueryNames& names, Query& q) : N(names), q(q) {}
	std::string label[MAX_NODES];  // the name as written (structure pattern, biome, building) per node

	// parse `text` and attach it as a child of `parent` (an N_ALL node)
	void parseInto(int parent, const std::string& text) {
		tokenize(text);
		pos = 0;
		int e = parseOr(true, 0, 0);
		if (pos < toks.size()) fail("unexpected '" + toks[pos].s + "'");
		if (q.n[e].kind == N_ALL && q.n[parent].kind == N_ALL) moveKids(parent, e);  // "a, b" at top level: no extra group
		else attach(parent, e);
	}

	int newNode(int kind) {
		if (q.nnodes >= MAX_NODES) throw std::runtime_error("query too large (max " + std::to_string(MAX_NODES) + " blocks)");
		Node& n = q.n[q.nnodes];
		memset(&n, 0, sizeof n);
		n.kind = kind;
		n.first = n.next = -1;
		n.count = 1;
		return q.nnodes++;
	}
	void moveKids(int to, int from) {  // children of a { } group become direct children of `to`
		for (int k = q.n[from].first; k >= 0;) {
			int nx = q.n[k].next;
			q.n[k].next = -1;
			attach(to, k);
			k = nx;
		}
		q.n[from].first = -1;
	}
	void attach(int parent, int child) {
		int* link = &q.n[parent].first;
		while (*link >= 0) link = &q.n[*link].next;
		*link = child;
	}

	// structure name -> masks; "" if it isn't a structure name
	bool structureMasks(std::string name, uint64_t& sm, uint32_t& setm) const {
		const Pack& P = *N.P;
		sm = 0, setm = 0;
		bool glob = name.find('*') != std::string::npos;
		auto globMatch = [](const std::string& pat, const std::string& s) {
			// '*' wildcard match
			size_t p = 0, si = 0, star = std::string::npos, mark = 0;
			while (si < s.size()) {
				if (p < pat.size() && pat[p] == s[si]) p++, si++;
				else if (p < pat.size() && pat[p] == '*') star = p++, mark = si;
				else if (star != std::string::npos) p = star + 1, si = ++mark;
				else return false;
			}
			while (p < pat.size() && pat[p] == '*') p++;
			return p == pat.size();
		};
		for (size_t si = 0; si < P.sets.size(); si++) {
			bool setMatch = !glob && (P.sets[si].name == name || P.sets[si].name == name + "s");
			for (int st : P.sets[si].structs) {
				std::string sn = P.structures[st].name.substr(P.structures[st].name.find(':') + 1);
				if (setMatch || (glob ? globMatch(name, sn) : sn == name)) {
					sm |= 1ULL << st;
					setm |= 1u << si;
				}
			}
		}
		return sm != 0;
	}

private:
	std::string buildingList(uint64_t ctx) const {
		std::vector<std::string> keys;
		for (int st = 0; st < 64; st++)
			if ((ctx >> st) & 1 && st < (int)N.buildings.size())
				for (auto& b : N.buildings[st]) {
					std::string k = buildingKey(b.first);
					if (std::find(keys.begin(), keys.end(), k) == keys.end()) keys.push_back(k);
				}
		std::sort(keys.begin(), keys.end());
		std::string s = " (has: ";
		for (size_t i = 0; i < keys.size(); i++) s += (i ? ", " : "") + keys[i];
		return s + ")";
	}
	std::string suggest(const std::string& name) const {
		std::vector<std::string> hits;
		for (auto& set : N.P->sets) {
			if (set.name.find(name) != std::string::npos) hits.push_back(set.name);
			for (int st : set.structs) {
				std::string sn = N.P->structures[st].name.substr(N.P->structures[st].name.find(':') + 1);
				if (sn.find(name) != std::string::npos && sn != set.name) hits.push_back(sn);
			}
		}
		if (hits.empty()) return " (seedwright <pack> list shows all names)";
		std::string s = "; did you mean ";
		for (size_t i = 0; i < hits.size() && i < 6; i++) s += (i ? ", " : "") + hits[i];
		return s + (hits.size() > 6 ? ", ...?" : "?") + " ('*" + name + "*' matches all of them)";
	}
	struct Tok {
		std::string s;
		size_t at;
	};
	const QueryNames& N;
	Query& q;
	std::vector<Tok> toks;
	size_t pos = 0;
	std::string src;

	[[noreturn]] void fail(const std::string& msg) const {
		size_t at = pos < toks.size() ? toks[pos].at : src.size();
		throw std::runtime_error("query: " + msg + "\n  " + src + "\n  " + std::string(at, ' ') + "^");
	}

	void tokenize(const std::string& text) {
		src = text;
		toks.clear();
		for (size_t i = 0; i < text.size();) {
			char ch = text[i];
			if (isspace((unsigned char)ch)) { i++; continue; }
			if (strchr("{}(),;=", ch)) {
				toks.push_back({std::string(1, ch == ';' ? ',' : ch), i});
				i++;
				continue;
			}
			size_t j = i;
			while (j < text.size() && !isspace((unsigned char)text[j]) && !strchr("{}(),;=", text[j])) j++;
			std::string w = text.substr(i, j - i);
			for (auto& c : w) c = (char)tolower((unsigned char)c);
			if (w.rfind("minecraft:", 0) == 0) w = w.substr(10);
			toks.push_back({w, i});
			i = j;
		}
	}

	bool peek(const char* s) const { return pos < toks.size() && toks[pos].s == s; }
	bool accept(const char* s) {
		if (!peek(s)) return false;
		pos++;
		return true;
	}
	void expect(const char* s) {
		if (!accept(s)) fail(std::string("expected '") + s + "'");
	}
	std::string word() {
		if (pos >= toks.size()) fail("unexpected end of query");
		return toks[pos++].s;
	}
	static bool isInt(const std::string& s) {
		size_t i = s[0] == '-' ? 1 : 0;
		if (i >= s.size()) return false;
		for (; i < s.size(); i++)
			if (!isdigit((unsigned char)s[i])) return false;
		return true;
	}
	int number(const char* what) {
		if (pos >= toks.size() || !isInt(toks[pos].s)) fail(std::string("expected ") + what);
		long v = std::stol(toks[pos++].s);
		if (v < -1000000 || v > 1000000) fail(std::string(what) + " out of range");
		accept("b") || accept("blocks");
		return (int)v;
	}
	int within() {
		if (!accept("within")) fail("expected 'within <blocks>'");
		int r = number("a distance in blocks");
		if (r < 0) fail("distance must be >= 0");
		return r;
	}

	// commas: true = ',' means 'and' here; false = inside any{ }, where ',' separates alternatives
	int parseOr(bool commaIsAnd, int depth, uint64_t structCtx) {
		int a = parseAnd(commaIsAnd, depth, structCtx);
		if (!peek("or")) return a;
		int any = newNode(N_ANY);
		attach(any, a);
		while (accept("or")) attach(any, parseAnd(commaIsAnd, depth, structCtx));
		return any;
	}
	int parseAnd(bool commaIsAnd, int depth, uint64_t structCtx) {
		int a = parseUnary(depth, structCtx);
		if (!(peek("and") || (commaIsAnd && peek(",")))) return a;
		int all = newNode(N_ALL);
		attach(all, a);
		while (accept("and") || (commaIsAnd && accept(","))) {
			if (peek("}") || peek(")") || pos >= toks.size()) break;  // trailing comma
			attach(all, parseUnary(depth, structCtx));
		}
		return all;
	}
	int parseUnary(int depth, uint64_t structCtx) {
		if (accept("not")) {
			int n = newNode(N_NOT);
			attach(n, parseUnary(depth, structCtx));
			return n;
		}
		return parseAtom(depth, structCtx);
	}
	int block(int kind, int depth, uint64_t structCtx) {  // '{' or '}' into a group node
		int g = newNode(kind);
		expect("{");
		if (accept("}")) return g;
		if (kind == N_ANY) {
			do attach(g, parseOr(false, depth, structCtx));
			while (accept(","));
		} else {
			int e = parseOr(true, depth, structCtx);
			if (q.n[e].kind == N_ALL) moveKids(g, e);  // "{ a, b }": a and b directly, no inner group
			else attach(g, e);
		}
		expect("}");
		return g;
	}
	int lootNode(int count, uint64_t structCtx) {
		std::string name = word();
		if (structCtx == 0) { pos--; fail("loot only works inside a structure's { } (village, buried_treasure)"); }
		if (N.lootItems.empty()) { pos--; fail("this pack has no loot data (rebuild it with tools/build-packs.sh)"); }
		auto find = [&](const std::string& n) {
			for (size_t i = 0; i < N.lootItems.size(); i++)
				if (N.lootItems[i] == n || N.lootItems[i] == "minecraft:" + n) return (int)i;
			return -1;
		};
		int item = find(name);
		if (item < 0 && name.size() > 1 && name.back() == 's') item = find(name.substr(0, name.size() - 1));  // "diamonds"
		bool anyLoot = false, gives = false, jig = false;
		std::vector<std::string> can;
		for (int st = 0; st < 64 && st < (int)N.lootGives.size(); st++) {
			if (!((structCtx >> st) & 1) || N.lootGives[st].empty()) continue;
			anyLoot = true;
			jig |= N.lootJig[st] != 0;
			if (item >= 0 && N.lootGives[st][item]) gives = true;
			for (size_t i = 0; i < N.lootGives[st].size(); i++)
				if (N.lootGives[st][i] && std::find(can.begin(), can.end(), N.lootItems[i]) == can.end()) can.push_back(N.lootItems[i]);
		}
		if (!anyLoot) { pos--; fail("no loot data for this structure (loot works in buried treasure, and in villages on 26.3+ where their layouts are exact)"); }
		if (!gives) {
			std::sort(can.begin(), can.end());
			std::string list;
			for (auto& c : can) list += (list.empty() ? "" : ", ") + c.substr(c.find(':') + 1);
			pos--;
			fail("no '" + name + "' in this structure's chests (they can hold: " + list + ")");
		}
		int n = newNode(N_LOOT);
		label[n] = N.lootItems[item].substr(N.lootItems[item].find(':') + 1);
		q.n[n].count = count;
		q.n[n].item = item;
		q.n[n].lootJig = jig ? 1 : 0;
		if (jig) q.hasBuildings = q.hasVillageLoot = 1;  // village layouts, exact check like buildings
		return n;
	}
	int parseAtom(int depth, uint64_t structCtx) {
		if (depth > MAX_DEPTH) fail("nested too deep (max " + std::to_string(MAX_DEPTH) + ")");
		if (accept("(")) {
			int e = parseOr(true, depth, structCtx);
			expect(")");
			return e;
		}
		if (accept("all")) return block(N_ALL, depth, structCtx);
		if (accept("any")) return block(N_ANY, depth, structCtx);
		if (accept("biome")) {
			std::string name = word();
			int b = N.P->biome(name);
			if (b < 0) { pos--; fail("unknown biome '" + name + "'"); }
			int n = newNode(N_BIOME);
			label[n] = name;
			q.n[n].biome = b;
			q.n[n].radius = within();
			q.n[n].y = 64;
			if (accept("at")) {
				expect("y");
				expect("=");
				if (accept("any")) q.n[n].y = Y_ANY;
				else q.n[n].y = number("a y level or 'any'");
			}
			if (q.n[n].y != Y_ANY && (q.n[n].y < N.P->dev->minY || q.n[n].y >= N.P->dev->minY + N.P->dev->height)) fail("y outside the world");
			return n;
		}
		if (accept("spawn")) {
			int n = newNode(N_SPAWN);
			q.n[n].radius = within();
			accept("of") && (accept("origin") || accept("0"));
			q.needSpawn = 1;
			return n;
		}
		int count = 1;
		if (pos < toks.size() && isInt(toks[pos].s)) count = number("a count");
		if (accept("loot")) {
			if (pos < toks.size() && isInt(toks[pos].s)) count = number("a count");
			if (count < 1 || count > 9999) fail("count must be 1..9999");
			return lootNode(count, structCtx);
		}
		if (count < 1 || count > 64) fail("count must be 1..64");
		bool isBuilding = accept("building") || accept("buildings");
		std::string name = word();
		if (!isBuilding && (name == "stronghold" || name == "strongholds")) {
			int n = newNode(N_STRONGHOLD);
			q.n[n].count = count;
			if (count > MAX_STRONGHOLDS) fail("at most " + std::to_string(MAX_STRONGHOLDS) + " strongholds");
			q.n[n].radius = within();
			if (peek("{")) moveKids(n, block(N_ALL, depth + 1, 0));
			return n;
		}
		uint64_t sm;
		uint32_t setm;
		if (!isBuilding && !structureMasks(name, sm, setm) && name.size() > 1 && name.back() == 's')
			structureMasks(name.substr(0, name.size() - 1), sm, setm);  // plural: "monuments", "igloos"
		if (!isBuilding && sm) {
			int n = newNode(N_STRUCTURE);
			label[n] = name;
			q.n[n].count = count;
			q.n[n].structMask = sm;
			q.n[n].setMask = setm;
			q.n[n].radius = within();
			if (peek("{")) moveKids(n, block(N_ALL, depth + 1, sm));  // the structure itself is an implicit ALL
			return n;
		}
		// a building of the enclosing village(s)
		if (structCtx == 0) { pos--; fail("unknown structure '" + name + "'" + (isBuilding ? " (buildings only work inside a village's { })" : suggest(name))); }
		uint64_t cat[3] = {0, 0, 0};
		bool anyLayouts = false;
		for (int pass = 0; pass < 2 && !(cat[0] | cat[1] | cat[2]); pass++) {  // pass 1: plural ("2 blacksmiths")
			if (pass == 1 && (name.size() < 2 || name.back() != 's')) break;
			std::string key = buildingKey(pass ? name.substr(0, name.size() - 1) : name);
			for (int st = 0; st < 64; st++)
				if ((structCtx >> st) & 1 && st < (int)N.buildings.size())
					for (auto& b : N.buildings[st]) {
						anyLayouts = true;
						if (buildingKey(b.first) == key) cat[b.second >> 6] |= 1ULL << (b.second & 63);
					}
		}
		if (!anyLayouts) { pos--; fail("'" + name + "' is not a structure, and this structure has no building layouts"); }
		if (!(cat[0] | cat[1] | cat[2])) { pos--; fail("no building '" + name + "' in this structure" + buildingList(structCtx)); }
		q.hasBuildings = 1;
		int n = newNode(N_BUILDING);
		label[n] = name;
		q.n[n].count = count;
		memcpy(q.n[n].cat, cat, sizeof cat);
		return n;
	}
};

// canonical text for a node (echoed back so the user sees what was understood)
inline std::string queryText(const Pack& P, const Query& q, int ni, const QueryNames& N, const std::string* label = nullptr) {
	const Node& n = q.n[ni];
	auto kids = [&](const char* sep) {
		std::string s;
		for (int k = n.first; k >= 0; k = q.n[k].next) s += (s.empty() ? "" : sep) + queryText(P, q, k, N, label);
		return s;
	};
	auto braces = [&]() { return n.first >= 0 ? " { " + kids(", ") + " }" : std::string(); };
	auto structName = [&]() -> std::string {
		if (label && !label[ni].empty()) return label[ni];
		for (size_t si = 0; si < P.sets.size(); si++) {
			uint64_t all = 0;
			for (int st : P.sets[si].structs) all |= 1ULL << st;
			if (all == n.structMask && P.sets[si].structs.size() > 1) return P.sets[si].name;
		}
		std::string s;
		for (int st = 0; st < 64; st++)
			if ((n.structMask >> st) & 1) s += (s.empty() ? "" : "|") + P.structures[st].name.substr(P.structures[st].name.find(':') + 1);
		return s;
	};
	switch (n.kind) {
	case N_ALL: return ni == q.root ? kids(", ") : "all { " + kids(", ") + " }";
	case N_ANY: return "any { " + kids(", ") + " }";
	case N_NOT: return "not " + (n.first >= 0 && q.n[n.first].next < 0 ? queryText(P, q, n.first, N, label) : "all { " + kids(", ") + " }");
	case N_STRUCTURE: return std::to_string(n.count) + " " + structName() + " within " + std::to_string(n.radius) + braces();
	case N_BIOME:
		return "biome " + P.biomes[n.biome].substr(10) + " within " + std::to_string(n.radius) +
		       (n.y == 64 ? "" : n.y == Y_ANY ? " at y=any" : " at y=" + std::to_string(n.y));
	case N_STRONGHOLD: return std::to_string(n.count) + " stronghold within " + std::to_string(n.radius) + braces();
	case N_SPAWN: return "spawn within " + std::to_string(n.radius);
	case N_BUILDING: return std::to_string(n.count) + " " + (label && !label[ni].empty() ? label[ni] : "building");
	case N_LOOT:
		return "loot " + std::to_string(n.count) + " " +
		       (label && !label[ni].empty() ? label[ni] : n.item < (int)N.lootItems.size() ? N.lootItems[n.item].substr(N.lootItems[n.item].find(':') + 1) : "item");
	}
	return "?";
}

// JSON tree for front ends: {"kind": ..., "name": ..., "count": .., "radius": .., "y": .., "children": [..]}
inline std::string queryJson(const Pack& P, const Query& q, int ni, const QueryNames& N, const std::string* label) {
	const Node& n = q.n[ni];
	static const char* kinds[] = {"all", "any", "not", "structure", "biome", "stronghold", "spawn", "building", "loot"};
	auto esc = [](const std::string& s) {
		std::string o;
		for (char c : s) o += (c == '"' || c == '\\') ? std::string("\\") + c : std::string(1, c);
		return o;
	};
	std::string j = std::string("{\"kind\": \"") + kinds[n.kind] + "\"";
	if (n.kind == N_STRUCTURE || n.kind == N_BUILDING) {
		std::string nm = label && !label[ni].empty() ? label[ni] : "";
		if (nm.empty()) {  // legacy flag without a label: canonical text minus count/radius
			std::string t = queryText(P, q, ni, N, label);
			nm = t.substr(t.find(' ') + 1);
			nm = nm.substr(0, nm.find(' '));
		}
		j += ", \"name\": \"" + esc(nm) + "\"";
	}
	if (n.kind == N_BIOME) j += ", \"name\": \"" + esc(P.biomes[n.biome].substr(10)) + "\", \"y\": " + (n.y == Y_ANY ? std::string("\"any\"") : std::to_string(n.y));
	if (n.kind == N_LOOT)
		j += ", \"name\": \"" + esc(n.item < (int)N.lootItems.size() ? N.lootItems[n.item].substr(N.lootItems[n.item].find(':') + 1) : "?") + "\"";
	if (n.kind == N_STRUCTURE || n.kind == N_STRONGHOLD || n.kind == N_BUILDING || n.kind == N_LOOT) j += ", \"count\": " + std::to_string(n.count);
	if (n.kind == N_STRUCTURE || n.kind == N_STRONGHOLD || n.kind == N_BIOME || n.kind == N_SPAWN) j += ", \"radius\": " + std::to_string(n.radius);
	if (n.kind == N_ALL || n.kind == N_ANY || n.kind == N_NOT || n.kind == N_STRUCTURE || n.kind == N_STRONGHOLD) {
		j += ", \"children\": [";
		bool first = true;
		for (int k = n.first; k >= 0; k = q.n[k].next) {
			j += (first ? "" : ", ") + queryJson(P, q, k, N, label);
			first = false;
		}
		j += "]";
	}
	return j + "}";
}

}  // namespace sw

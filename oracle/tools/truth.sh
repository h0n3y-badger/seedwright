#!/usr/bin/env bash
# Headless ground truth for one or more seeds: fresh dev world per seed, dedicated server, no GUI.
# MC=<version> selects the game version (default: gradle.properties); 1.21.1-26.2 use the legacy oracle.
# usage: tools/truth.sh "radius=40 points=4000" seed1 [seed2 ...]   -> ../truth/<mc>/oracle-<seed>.txt
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mc=${MC:-$(grep '^minecraft_version=' "$root/gradle.properties" | cut -d= -f2)}
extra=(-Pminecraft_version="$mc")
proj="$root"
case "$mc" in
	26.3*) ;;
	1.*) proj="$root/../oracle-legacy" ;;  # obfuscated: remapping Loom
	*) extra+=(-Plegacy) ;;
esac
opts="$1"; shift
export JAVA_HOME=/usr/lib/jvm/java-25-openjdk
out="$root/../truth/$mc"
root="$proj"
mkdir -p "$root/run" "$out"
echo "eula=true" >| "$root/run/eula.txt"
for seed in "$@"; do
	rm -rf -- "$root/run/world"
	printf 'level-seed=%s\nonline-mode=false\nspawn-protection=0\nsync-chunk-writes=false\n' "$seed" >| "$root/run/server.properties"
	"$root/gradlew" -p "$root" runServer "${extra[@]}" -Poracle="$opts" --args=nogui --console=plain > "$root/run-server-$seed.log" 2>&1 || true
	f=$(ls -t "$root"/run/oracle-*.txt 2>/dev/null | head -1)
	[[ -n "$f" ]] && mv -f "$f" "$out/" && echo "ok $seed"
	f=$(ls -t "$root"/run/villages-*.txt 2>/dev/null | head -1 || true)  # only with villages=R in the options
	[[ -n "$f" ]] && mv -f "$f" "$out/" && echo "ok $seed villages"
	cp -f "$root/run/biome_parameters.txt" "$out/" 2>/dev/null || true
done

#!/usr/bin/env bash
# Headless ground truth for one or more seeds: fresh dev world per seed, dedicated server, no GUI.
# usage: tools/truth.sh "radius=40 points=4000" seed1 [seed2 ...]   -> ../truth/<mc>/oracle-<seed>.txt
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mc=$(grep '^minecraft_version=' "$root/gradle.properties" | cut -d= -f2)
opts="$1"; shift
export JAVA_HOME=/usr/lib/jvm/java-25-openjdk
mkdir -p "$root/run" "$root/../truth/$mc"
echo "eula=true" > "$root/run/eula.txt"
for seed in "$@"; do
	rm -rf -- "$root/run/world"
	printf 'level-seed=%s\nonline-mode=false\nspawn-protection=0\nsync-chunk-writes=false\n' "$seed" > "$root/run/server.properties"
	"$root/gradlew" -p "$root" runServer -Poracle="$opts" --args=nogui --console=plain > "$root/run-server-$seed.log" 2>&1 || true
	f=$(ls -t "$root"/run/oracle-*.txt 2>/dev/null | head -1)
	[[ -n "$f" ]] && mv "$f" "$root/../truth/$mc/" && echo "ok $seed"
	cp "$root/run/biome_parameters.txt" "$root/../truth/$mc/" 2>/dev/null || true
done

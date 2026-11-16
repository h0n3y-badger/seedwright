#!/usr/bin/env bash
# Build packs/<version>.pack from vanilla jars you already have (the game's data isn't redistributed here).
# usage: tools/build-packs.sh [jar-dir] [version...]
#   jar-dir: directory with <version>.jar (client or server jar), default $JARS or ~/clones/mc-jars
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
jars=${1:-${JARS:-$HOME/clones/mc-jars}}
shift || true
versions=("$@")
[ ${#versions[@]} -gt 0 ] || versions=($(ls "$root/truth"))
mkdir -p "$root/packs"
for v in "${versions[@]}"; do
	if [ ! -f "$jars/$v.jar" ]; then echo "$v: no $jars/$v.jar, skipped"; continue; fi
	python3 "$root/tools/extract.py" "$jars/$v.jar" "$v" "$root/truth/$v/biome_parameters.txt" "$root/packs/$v.pack" > /dev/null
	echo "$v: packs/$v.pack"
done

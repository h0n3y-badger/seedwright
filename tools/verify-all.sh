#!/usr/bin/env bash
# Rebuild every version pack from its jar + oracle biome list, then verify against the oracle truth.
set -u
root="$(cd "$(dirname "$0")/.." && pwd)"
jars=${JARS:-$HOME/clones/mc-jars}
for v in ${VERSIONS:-1.21.1 1.21.2 1.21.3 1.21.4 1.21.5 1.21.6 1.21.7 1.21.8 1.21.9 1.21.10 1.21.11 26.1 26.1.1 26.1.2 26.2 26.3}; do
	python3 "$root/tools/extract.py" "$jars/$v.jar" "$v" "$root/truth/$v/biome_parameters.txt" "$root/packs/$v.pack" > /dev/null || { echo "$v: EXTRACT FAILED"; continue; }
	out=$("$root/engine/seedwright-cpu" "$root/packs/$v.pack" verify "$root"/truth/$v/oracle-*.txt 2>&1)
	starts=$(echo "$out" | grep 'structure starts' | sed 's/.*height): //' | tr '\n' ';')
	echo "$v: $(echo "$out" | tail -1) | starts: $starts"
done

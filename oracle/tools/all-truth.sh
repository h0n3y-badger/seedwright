#!/usr/bin/env bash
# ground truth for every supported version (headless); logs per version in ../truth/<v>/run.log
set -u
here="$(cd "$(dirname "$0")" && pwd)"
for v in ${VERSIONS:-1.21.1 1.21.2 1.21.3 1.21.4 1.21.5 1.21.6 1.21.7 1.21.8 1.21.9 1.21.10 1.21.11 26.1 26.1.1 26.1.2 26.2}; do
	mkdir -p "$here/../../truth/$v"
	MC=$v "$here/truth.sh" "radius=40" 12345 -4172144997902289642 >| "$here/../../truth/$v/run.log" 2>&1
	echo "$v: $(ls "$here/../../truth/$v" | grep -c oracle-) seeds"
done

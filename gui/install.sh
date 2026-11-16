#!/bin/sh
# Install (or with --uninstall, remove) the seedwright GUI launcher for the current user.
# The .desktop file points at this checkout, so re-run it if you move the repo.
set -eu
GUI_DIR=$(cd "$(dirname "$0")" && pwd)
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DEST="$APPS/seedwright-gui.desktop"

if [ "${1:-}" = "--uninstall" ]; then
	rm -f "$DEST"
	echo "removed $DEST"
else
	mkdir -p "$APPS"
	sed "s|@GUI_DIR@|$GUI_DIR|g" "$GUI_DIR/seedwright-gui.desktop.in" >| "$DEST"
	chmod +x "$GUI_DIR/seedwright_gui.py"
	echo "installed $DEST -> $GUI_DIR/seedwright_gui.py"
fi
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database -q "$APPS" || true

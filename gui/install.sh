#!/bin/sh
# Install the seedwright GUI launcher.
#   gui/install.sh                    per user: ~/.local/share/applications (KDE menu etc.)
#   doas gui/install.sh --system      /usr/share/applications (needed for launchers that only read
#                                     that directory, e.g. tux)
#   add --uninstall to either to remove it again.
# The .desktop file points at this checkout, so re-run it if you move the repo.
set -eu
GUI_DIR=$(cd "$(dirname "$0")" && pwd)
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
UNINSTALL=0
for a in "$@"; do
	case "$a" in
	--system) APPS=/usr/share/applications ;;
	--uninstall) UNINSTALL=1 ;;
	*) echo "usage: $0 [--system] [--uninstall]" >&2; exit 2 ;;
	esac
done
DEST="$APPS/seedwright-gui.desktop"

if [ "$UNINSTALL" = 1 ]; then
	rm -f "$DEST"
	echo "removed $DEST"
else
	mkdir -p "$APPS"
	sed "s|@GUI_DIR@|$GUI_DIR|g" "$GUI_DIR/seedwright-gui.desktop.in" >| "$DEST"
	chmod 644 "$DEST"
	echo "installed $DEST -> $GUI_DIR/seedwright_gui.py"
fi
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database -q "$APPS" || true

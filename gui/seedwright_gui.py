#!/usr/bin/env python3
"""seedwright GUI: Scratch-style blocks for seedwright's query language.

Drag blocks from the palette into the script; C-shaped blocks (structures, strongholds, all/any/none)
hold blocks that are measured from them ("a plains village within 500 that has 2 blacksmiths").
The blocks and the query text below them are two views of the same query: the text is what the CLI's
--where takes, and typed text can be turned back into blocks (the engine's `parse` command).

Run from anywhere: python3 gui/seedwright_gui.py
"""
import fnmatch
import itertools
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QMimeData, QPoint, QProcess, QRegularExpression, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QDrag, QFont, QGuiApplication, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QRadioButton,
    QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parent.parent
PACKS = ROOT / "packs"
ENGINE = ROOT / "engine"
MIME = "application/x-seedwright-block"
MAX_BLOCKS = 47  # query.h MAX_NODES minus the root

CAVE_BIOMES = {"lush_caves", "dripstone_caves", "deep_dark"}


def version_key(name):
    return [int(p) for p in re.findall(r"\d+", name)]


def engine_exe():
    for n in ("seedwright-cpu", "seedwright-gpu"):
        if (ENGINE / n).exists():
            return ENGINE / n
    return None


# ------------------------------------------------------------------------------------------ small widgets

class SliderEntry(QWidget):
    """Slider plus an editable spin box, kept in sync both ways (also while typing)."""
    valueChanged = Signal(int)

    def __init__(self, lo, hi, value, step=1, suffix="", width=None):
        super().__init__()
        self.step = step
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo // step, hi // step)
        if width:
            self.slider.setMinimumWidth(width)
        self.spin = QSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setSuffix(suffix)
        self.spin.setMinimumWidth(84)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.slider.valueChanged.connect(lambda v: self.spin.setValue(v * self.step))
        self.spin.valueChanged.connect(self._spin)
        self.spin.setValue(value)
        self._spin(value)

    def _spin(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(round(v / self.step))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)

    def value(self):
        return self.spin.value()

    def setValue(self, v):
        self.spin.setValue(int(v))


class Log2Entry(QWidget):
    """Slider over powers of two plus a free-form entry (plain integer or 2^N)."""
    valueChanged = Signal()

    def __init__(self, lo_exp, hi_exp, exp):
        super().__init__()
        self.lo_exp, self.hi_exp = lo_exp, hi_exp
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo_exp, hi_exp)
        self.edit = QLineEdit()
        self.edit.setMinimumWidth(110)
        self.edit.setToolTip("Integer, or 2^N")
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.edit)
        self.slider.valueChanged.connect(lambda e: (self.edit.setText(f"2^{e}"), self.valueChanged.emit()))
        self.edit.textEdited.connect(lambda _: self._edited(live=True))
        self.edit.editingFinished.connect(self._edited)
        self.edit.setText(f"2^{exp}")
        self.slider.setValue(exp)

    @staticmethod
    def parse(text):
        t = text.strip().replace(" ", "").replace("_", "").replace(",", "")
        m = re.fullmatch(r"2(?:\^|\*\*)(\d+)", t)
        if m:
            return 1 << int(m.group(1))
        return int(t) if t.isdigit() and int(t) > 0 else None

    def _edited(self, live=False):
        v = self.parse(self.edit.text())
        if v is None:
            if not live:
                self.edit.setText(f"2^{self.slider.value()}")
            return
        v = min(v, 1 << 48)
        self.slider.blockSignals(True)
        self.slider.setValue(max(self.lo_exp, min(self.hi_exp, v.bit_length() - 1)))
        self.slider.blockSignals(False)
        self.valueChanged.emit()

    def value(self):
        return self.parse(self.edit.text()) or (1 << self.slider.value())

    def text(self):
        return self.edit.text()

    def setText(self, t):
        self.edit.setText(t)
        self._edited()


def searchable_combo(min_width=170):
    c = QComboBox()
    c.setEditable(True)
    c.setInsertPolicy(QComboBox.NoInsert)
    c.setMinimumWidth(min_width)
    c.completer().setFilterMode(Qt.MatchContains)
    c.completer().setCompletionMode(c.completer().CompletionMode.PopupCompletion)
    return c


# ------------------------------------------------------------------------------------------ blocks

# kind: (palette label, colour, has a mouth, mouth caption)
KINDS = {
    "structure": ("structure", "#e6801a", True, "that has / with, measured from it:"),
    "biome": ("biome", "#3f9b46", False, ""),
    "stronghold": ("stronghold", "#7b5ad6", True, "with, measured from it:"),
    "spawn": ("spawn near 0,0", "#3d7fe0", False, ""),
    "all": ("all of", "#c99400", True, ""),
    "any": ("any of", "#c99400", True, ""),
    "not": ("none of", "#c4572f", True, ""),
    "building": ("village building", "#9a6a2e", False, ""),
}
CATEGORIES = [
    ("Structures", ["structure", "stronghold"]),
    ("Biomes", ["biome"]),
    ("Villages", ["building"]),
    ("Logic", ["all", "any", "not"]),
    ("World", ["spawn"]),
]
HELP = {
    "structure": "At least N of a structure within a distance. Blocks inside it are measured from that structure.",
    "biome": "A biome within a distance (16-block grid). Cave biomes need a depth: 'any cave depth' tries y -48..48.",
    "stronghold": "At least N strongholds within a distance (any ring). Blocks inside are measured from the stronghold.",
    "spawn": "The world spawn point is within this many blocks of 0,0.\n"
             "(Every other block is measured from spawn already; this one only matters if you want spawn near the "
             "world centre, e.g. for a map.)",
    "all": "Every block inside must hold.",
    "any": "At least one block inside must hold.",
    "not": "None of the blocks inside may hold.",
    "building": "Buildings in the village around it (put it inside a village block). weaponsmith = blacksmith.\n"
                "26.3: exact (every result is re-checked with the game's terrain heights).\n"
                "Older versions (≈): layouts use estimated terrain height, so counts are plausible, not guaranteed.",
}


class Mouth(QFrame):
    """The inside of a C-block (or the script root): a vertical stack that accepts dropped blocks."""
    changed = Signal()

    def __init__(self, app, owner, caption=""):
        super().__init__()
        self.app, self.owner = app, owner
        self.setAcceptDrops(True)
        self.setObjectName("mouth")
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(10, 4, 4, 6)
        self.lay.setSpacing(4)
        self.caption = QLabel(caption)
        self.caption.setObjectName("caption")
        self.caption.setVisible(bool(caption))
        self.lay.addWidget(self.caption)
        self.empty = QLabel("drop blocks here")
        self.empty.setObjectName("empty")
        self.lay.addWidget(self.empty)
        self.lay.addStretch()

    def blocks(self):
        out = []
        for i in range(self.lay.count()):
            w = self.lay.itemAt(i).widget()
            if isinstance(w, Block):
                out.append(w)
        return out

    def insert(self, block, index=None):
        bl = self.blocks()
        if index is None or index > len(bl):
            index = len(bl)
        # layout index: caption, empty label, then blocks (the indicator may sit in between)
        pos = self.lay.indexOf(bl[index]) if index < len(bl) else self.lay.count() - 1
        self.lay.insertWidget(pos, block)
        block.show()
        self.empty.setVisible(False)

    def take(self, block):
        self.lay.removeWidget(block)
        self.empty.setVisible(not self.blocks())

    def dropIndex(self, y):
        for i, b in enumerate(self.blocks()):
            if y < b.geometry().center().y():
                return i
        return len(self.blocks())

    def _accepts(self, e):
        if not e.mimeData().hasFormat(MIME):
            return False
        src = bytes(e.mimeData().data(MIME)).decode()
        if src.startswith("move:"):
            b = self.app.blockById(int(src[5:]))
            if b is None or b is self.owner or (self.owner is not None and b.isAncestorOf(self)):
                return False
        return True

    def dragEnterEvent(self, e):
        if self._accepts(e):
            e.acceptProposedAction()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        if not self._accepts(e):
            e.ignore()
            return
        e.acceptProposedAction()
        self.app.showIndicator(self, self.dropIndex(e.position().toPoint().y()))

    def dragLeaveEvent(self, e):
        self.app.hideIndicator()

    def dropEvent(self, e):
        self.app.hideIndicator()
        if not self._accepts(e):
            return
        idx = self.dropIndex(e.position().toPoint().y())
        src = bytes(e.mimeData().data(MIME)).decode()
        if src.startswith("new:"):
            if self.app.blockCount() >= MAX_BLOCKS:
                self.app.setStatus(f"at most {MAX_BLOCKS} blocks")
                return
            self.insert(self.app.makeBlock({"kind": src[4:]}), idx)
        else:
            b = self.app.blockById(int(src[5:]))
            old = b.parentMouth()
            if old is self:
                cur = self.blocks().index(b)
                if idx > cur:
                    idx -= 1
            old.take(b)
            self.insert(b, idx)
        e.acceptProposedAction()
        self.app.queryChanged()


class Block(QFrame):
    ids = itertools.count(1)

    def __init__(self, app, kind):
        super().__init__()
        self.app, self.kind = app, kind
        self.id = next(Block.ids)
        label, color, hasMouth, caption = KINDS[kind]
        self.color = color
        self.setObjectName("block")
        self.setToolTip(HELP[kind])
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 6, 6)
        v.setSpacing(4)
        self.head = QGridLayout()
        self.head.setHorizontalSpacing(6)
        self.head.setVerticalSpacing(3)
        v.addLayout(self.head)
        self.warn = QLabel()
        self.warn.setObjectName("warn")
        self.warn.setWordWrap(True)
        self.warn.hide()
        v.addWidget(self.warn)
        self.mouth = Mouth(app, self, caption) if hasMouth else None
        if self.mouth:
            v.addWidget(self.mouth)
        self.dragStart = None
        self.build()
        self.restyle(False)

    # ---- appearance
    def restyle(self, bad):
        c = QColor(self.color)
        border = "#ff4040" if bad else c.darker(140).name()
        self.setStyleSheet(f"""
            QFrame#block {{ background: {c.name()}; border: 2px solid {border}; border-radius: 9px; }}
            QFrame#block > QLabel, QFrame#block QLabel#title {{ color: white; font-weight: 600; background: transparent; }}
            QLabel#warn {{ color: #fff3c4; font-weight: 400; }}
            QFrame#mouth {{ background: rgba(0,0,0,0.28); border: none; border-radius: 6px; }}
            QLabel#caption {{ color: rgba(255,255,255,0.85); font-style: italic; background: transparent; }}
            QLabel#empty {{ color: rgba(255,255,255,0.55); background: transparent; }}
        """)

    def title(self, text):
        l = QLabel(text)
        l.setObjectName("title")
        return l

    def build(self):
        k, h = self.kind, self.head
        rm = QPushButton("✕")
        rm.setFixedSize(26, 24)
        rm.setToolTip("Remove this block")
        rm.clicked.connect(lambda: self.app.removeBlock(self))
        grip = self.title("⠿")
        grip.setToolTip("Drag to move")
        h.addWidget(grip, 0, 0)
        cols = 6
        h.addWidget(rm, 0, cols)
        h.setColumnStretch(3, 1)
        ch = self.app.queryChanged
        if k in ("structure", "stronghold", "building"):
            self.count = SliderEntry(1, 32 if k == "stronghold" else 16, 1, width=50)
            self.count.valueChanged.connect(ch)
            h.addWidget(self.title("at least"), 0, 1)
            h.addWidget(self.count, 0, 2)
        if k == "structure":
            self.target = searchable_combo()
            self.target.setToolTip("A structure, a whole set ('any'), or a pattern like village_* or *ruin*")
            self.target.currentTextChanged.connect(ch)
            h.addWidget(self.target, 0, 3, 1, cols - 3)
        elif k == "stronghold":
            h.addWidget(self.title("stronghold(s)"), 0, 3)
        elif k == "building":
            self.target = searchable_combo(170)
            self.target.currentTextChanged.connect(ch)
            h.addWidget(self.target, 0, 3, 1, 2)
            self.approx = self.title("≈")
            self.approx.setToolTip(HELP["building"])
            h.addWidget(self.approx, 0, 5)
        elif k == "biome":
            h.addWidget(self.title("biome"), 0, 1)
            self.target = searchable_combo()
            self.target.currentTextChanged.connect(self._biomePicked)
            h.addWidget(self.target, 0, 2, 1, cols - 2)
        elif k == "spawn":
            h.addWidget(self.title("world spawn is within"), 0, 1, 1, 2)
        elif k in ("all", "any", "not"):
            h.addWidget(self.title({"all": "all of these", "any": "any of these", "not": "none of these"}[k]), 0, 1, 1, 3)
        if k in ("structure", "stronghold", "biome", "spawn"):
            self.radius = SliderEntry(0, 20000, {"stronghold": 3000, "biome": 1000}.get(k, 500), step=16, suffix=" b", width=110)
            self.radius.valueChanged.connect(ch)
            row = 0 if k == "spawn" else 1
            if k != "spawn":
                self.withinLabel = self.title("within")
                h.addWidget(self.withinLabel, row, 1)
            h.addWidget(self.radius, row, 2 if k != "spawn" else 3, 1, 3 if k != "spawn" else 2)
            self.ofLabel = self.title("of 0,0" if k == "spawn" else "of spawn")
            h.addWidget(self.ofLabel, row, 5)
        if k == "biome":
            h.addWidget(self.title("at"), 2, 1)
            self.height = QComboBox()
            self.height.addItems(["surface (y=64)", "any cave depth", "height y ="])
            self.height.currentIndexChanged.connect(self._heightMode)
            self.y = SliderEntry(-64, 319, 0, width=120)
            self.y.valueChanged.connect(ch)
            h.addWidget(self.height, 2, 2)
            h.addWidget(self.y, 2, 3, 1, 3)
            self._heightMode()

    def _heightMode(self, *_):
        self.y.setVisible(self.height.currentIndex() == 2)
        self.app.queryChanged()

    def _biomePicked(self, *_):
        if self.target.currentText().strip() in CAVE_BIOMES and self.height.currentIndex() == 0:
            self.height.setCurrentIndex(1)  # caves don't exist at y=64
        self.app.queryChanged()

    # ---- tree helpers
    def parentMouth(self):
        p = self.parentWidget()
        while p is not None and not isinstance(p, Mouth):
            p = p.parentWidget()
        return p

    def parentBlock(self):
        m = self.parentMouth()
        return m.owner if m else None

    def enclosingStructure(self):
        b = self.parentBlock()
        while b is not None and b.kind != "structure":
            b = b.parentBlock()
        return b

    # ---- names (filled per version)
    def fillNames(self, names):
        if self.kind == "structure":
            keep = self.targetValue() or "villages"
            self.target.blockSignals(True)
            self.target.clear()
            for s in names["sets"]:
                structs = names["structs"][s]
                self.target.addItem(f"{s}  (any)" if len(structs) > 1 else structs[0], s if len(structs) > 1 else structs[0])
                if len(structs) > 1:
                    for st in structs:
                        self.target.addItem(f"    {st}", st)
            self.target.blockSignals(False)
            self.setTarget(keep)
        elif self.kind == "biome":
            keep = self.targetValue() or "cherry_grove"
            self.target.blockSignals(True)
            self.target.clear()
            for b in sorted(names["biomes"]):
                self.target.addItem(b, b)
            self.target.blockSignals(False)
            self.setTarget(keep)

    def fillBuildings(self, options):
        keep = self.targetValue() or "weaponsmith"
        self.target.blockSignals(True)
        self.target.clear()
        for b in options:
            self.target.addItem(b, b)
        self.target.blockSignals(False)
        self.setTarget(keep)

    def setTarget(self, value):
        i = self.target.findData(value)
        if i < 0 and value == "blacksmith":
            i = self.target.findData("weaponsmith")
        if i >= 0:
            self.target.setCurrentIndex(i)
        else:
            self.target.setEditText(value)

    def targetValue(self):
        if not hasattr(self, "target") or self.target.count() == 0 and not self.target.currentText():
            return ""
        txt = self.target.currentText().strip()
        i = self.target.currentIndex()
        if i >= 0 and self.target.itemText(i).strip() == txt:
            return self.target.itemData(i)
        return txt.replace("  (any)", "").split(" ")[0]

    # ---- query text / state
    def query(self):
        k = self.kind
        kids = ", ".join(b.query() for b in self.mouth.blocks()) if self.mouth else ""
        inner = f" {{ {kids} }}" if kids else ""
        if k == "structure":
            return f"{self.count.value()} {self.targetValue()} within {self.radius.value()}{inner}"
        if k == "stronghold":
            return f"{self.count.value()} strongholds within {self.radius.value()}{inner}"
        if k == "biome":
            y = ["", " at y=any", f" at y={self.y.value()}"][self.height.currentIndex()]
            return f"biome {self.targetValue()} within {self.radius.value()}{y}"
        if k == "spawn":
            return f"spawn within {self.radius.value()}"
        if k == "building":
            return f"{self.count.value()} {self.targetValue()}"
        if k == "all":
            return f"all {{ {kids} }}"
        if k == "any":
            return f"any {{ {kids} }}"
        return f"not all {{ {kids} }}" if len(self.mouth.blocks()) > 1 else (f"not {kids}" if kids else "not all { }")

    def state(self):
        s = {"kind": self.kind}
        if hasattr(self, "target"):
            s["name"] = self.targetValue()
        if hasattr(self, "count"):
            s["count"] = self.count.value()
        if hasattr(self, "radius"):
            s["radius"] = self.radius.value()
        if self.kind == "biome":
            s["y"] = [64, "any", self.y.value()][self.height.currentIndex()]
        if self.mouth:
            s["children"] = [b.state() for b in self.mouth.blocks()]
        return s

    def setState(self, s):
        if "count" in s and hasattr(self, "count"):
            self.count.setValue(s["count"])
        if "radius" in s and hasattr(self, "radius"):
            self.radius.setValue(s["radius"])
        if self.kind == "biome":
            y = s.get("y", 64)
            if y == "any":
                self.height.setCurrentIndex(1)
            elif y == 64:
                self.height.setCurrentIndex(0)
            else:
                self.height.setCurrentIndex(2)
                self.y.setValue(y)
        if "name" in s and hasattr(self, "target"):
            if self.kind == "building":
                self.target.setEditText(s["name"])
                self._pendingName = s["name"]
            else:
                self.setTarget(s["name"])
                if self.kind == "biome" and s.get("y") == 64:
                    self.height.setCurrentIndex(0)
        for c in s.get("children", []):
            self.mouth.insert(self.app.makeBlock(c))

    # ---- dragging
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.dragStart = e.position().toPoint()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self.dragStart is None or (e.position().toPoint() - self.dragStart).manhattanLength() < QApplication.startDragDistance():
            return
        self.dragStart = None
        drag = QDrag(self)
        md = QMimeData()
        md.setData(MIME, f"move:{self.id}".encode())
        drag.setMimeData(md)
        pm = self.grab()
        if pm.width() > 420:
            pm = pm.scaledToWidth(420, Qt.SmoothTransformation)
        drag.setPixmap(pm)
        drag.setHotSpot(QPoint(12, 12))
        drag.exec(Qt.MoveAction)

    def mouseReleaseEvent(self, e):
        self.dragStart = None
        super().mouseReleaseEvent(e)


class PaletteItem(QLabel):
    def __init__(self, app, kind):
        label, color, hasMouth, _ = KINDS[kind]
        super().__init__(("⊏ " if hasMouth else "▭ ") + label)
        self.app, self.kind = app, kind
        self.setToolTip(HELP[kind] + "\n\nDrag into the script (or click to add at the end).")
        c = QColor(color)
        self.setStyleSheet(f"QLabel {{ background: {c.name()}; color: white; font-weight: 600; border: 2px solid {c.darker(140).name()};"
                           f" border-radius: 8px; padding: 6px 10px; }}")
        self.setCursor(Qt.OpenHandCursor)
        self.dragStart = None

    def mousePressEvent(self, e):
        self.dragStart = e.position().toPoint()

    def mouseMoveEvent(self, e):
        if self.dragStart is None or (e.position().toPoint() - self.dragStart).manhattanLength() < QApplication.startDragDistance():
            return
        self.dragStart = None
        drag = QDrag(self)
        md = QMimeData()
        md.setData(MIME, f"new:{self.kind}".encode())
        drag.setMimeData(md)
        drag.setPixmap(self.grab())
        drag.exec(Qt.CopyAction | Qt.MoveAction)

    def mouseReleaseEvent(self, e):
        if self.dragStart is not None:
            self.app.addBlock({"kind": self.kind})
        self.dragStart = None


class Palette(QWidget):
    """Block palette; dropping a block from the script here deletes it."""

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setAcceptDrops(True)
        v = QVBoxLayout(self)
        v.setSpacing(6)
        self.items = {}
        for title, kinds in CATEGORIES:
            t = QLabel(title)
            t.setStyleSheet("font-weight: 700; margin-top: 6px;")
            v.addWidget(t)
            for k in kinds:
                it = PaletteItem(app, k)
                self.items[k] = it
                v.addWidget(it)
        v.addStretch()
        hint = QLabel("Drop a block here to delete it.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray;")
        v.addWidget(hint)

    def dragEnterEvent(self, e):
        if e.mimeData().hasFormat(MIME) and bytes(e.mimeData().data(MIME)).decode().startswith("move:"):
            e.acceptProposedAction()

    def dropEvent(self, e):
        b = self.app.blockById(int(bytes(e.mimeData().data(MIME)).decode()[5:]))
        if b:
            self.app.removeBlock(b)
        e.acceptProposedAction()


# ------------------------------------------------------------------------------------------ main window

class CrackPanel(QWidget):
    """'Crack a seed' mode: structures seen in a world (+ the spawn point) -> `seedwright crack`."""

    def __init__(self, main):
        super().__init__()
        self.main = main
        v = QVBoxLayout(self)
        intro = QLabel(
            "Type in structures you have found in a world, by the coordinates <b>/locate</b> gives (any block in the "
            "structure's start chunk). Structure positions fix the lower 48 bits of the seed, the structure types "
            "(biomes) and the spawn point fix the rest.<br>About <b>5-7 structures</b> usually give a single seed; "
            "villages, temples, outposts, shipwrecks, ruined portals, monuments and trial chambers all work. "
            "Mineshafts and strongholds don't.")
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        v.addWidget(intro)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Structure", "X", "Z", ""])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c, wd in ((1, 140), (2, 140), (3, 36)):
            hh.setSectionResizeMode(c, QHeaderView.Fixed)
            self.table.setColumnWidth(c, wd)
        self.table.verticalHeader().hide()
        v.addWidget(self.table, 1)
        row = QHBoxLayout()
        add = QPushButton("+ Add structure")
        add.clicked.connect(lambda: self.addClue())
        row.addWidget(add)
        row.addStretch()
        self.info = QLabel()
        row.addWidget(self.info)
        v.addLayout(row)
        sp = QHBoxLayout()
        self.useSpawn = QCheckBox("World spawn at  X")
        self.spX, self.spZ = QSpinBox(), QSpinBox()
        for w in (self.spX, self.spZ):
            w.setRange(-30_000_000, 30_000_000)
        self.spTol = QSpinBox()
        self.spTol.setRange(0, 1024)
        self.spTol.setValue(128)
        self.spTol.setSuffix(" b")
        self.spTol.setToolTip("How far the spawn point may be from where seedwright puts the spawn search origin\n"
                              "(the game moves the actual spawn up to ~100 blocks to find a grass block)")
        sp.addWidget(self.useSpawn)
        sp.addWidget(self.spX)
        sp.addWidget(QLabel("Z"))
        sp.addWidget(self.spZ)
        sp.addWidget(QLabel("give or take"))
        sp.addWidget(self.spTol)
        sp.addStretch()
        v.addLayout(sp)
        for w in (self.useSpawn,):
            w.toggled.connect(self.changed)
        for w in (self.spX, self.spZ, self.spTol):
            w.valueChanged.connect(self.changed)

    def structureNames(self):
        n = self.main.names
        return sorted({st for s in n["sets"] if s not in ("strongholds", "mineshafts", "buried_treasures") for st in n["structs"][s]})

    def addClue(self, name="village_plains", x=0, z=0):
        r = self.table.rowCount()
        self.table.insertRow(r)
        combo = searchable_combo(200)
        combo.addItems(self.structureNames())
        combo.setCurrentText(name)
        combo.currentTextChanged.connect(self.changed)
        self.table.setCellWidget(r, 0, combo)
        for c, val in ((1, x), (2, z)):
            sb = QSpinBox()
            sb.setRange(-30_000_000, 30_000_000)
            sb.setValue(val)
            sb.valueChanged.connect(self.changed)
            self.table.setCellWidget(r, c, sb)
        rm = QPushButton("✕")
        rm.setFixedWidth(28)
        rm.clicked.connect(lambda: self.removeClue(rm))
        self.table.setCellWidget(r, 3, rm)
        self.changed()

    def removeClue(self, btn):
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 3) is btn:
                self.table.removeRow(r)
                break
        self.changed()

    def refreshNames(self):
        names = self.structureNames()
        for r in range(self.table.rowCount()):
            c = self.table.cellWidget(r, 0)
            cur = c.currentText()
            c.blockSignals(True)
            c.clear()
            c.addItems(names)
            c.setCurrentText(cur)
            c.blockSignals(False)

    def clues(self):
        return [(self.table.cellWidget(r, 0).currentText(), self.table.cellWidget(r, 1).value(), self.table.cellWidget(r, 2).value())
                for r in range(self.table.rowCount())]

    def args(self):
        a = [f"{n}@{x},{z}" for n, x, z in self.clues()]
        if self.useSpawn.isChecked():
            a.append(f"spawn@{self.spX.value()},{self.spZ.value()}~{self.spTol.value()}")
        return a

    def changed(self, *_):
        n = self.table.rowCount()
        self.info.setText(f"{n} structure(s)" + ("  — usually not enough yet" if n < 5 else ""))
        self.main.queryChanged()

    def state(self):
        return {"clues": self.clues(), "spawn": self.useSpawn.isChecked(), "spX": self.spX.value(), "spZ": self.spZ.value(),
                "tol": self.spTol.value()}

    def setState(self, s):
        self.table.setRowCount(0)
        for n, x, z in s.get("clues", []):
            self.addClue(n, x, z)
        self.useSpawn.setChecked(bool(s.get("spawn")))
        self.spX.setValue(s.get("spX", 0))
        self.spZ.setValue(s.get("spZ", 0))
        self.spTol.setValue(s.get("tol", 128))


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("seedwright")
        self.settings = QSettings("seedwright", "gui")
        self.proc = None
        self.names = {"sets": [], "structs": {}, "biomes": [], "buildings": {}}
        self.loading = False
        self.indicator = QFrame()
        self.indicator.setFixedHeight(5)
        self.indicator.setStyleSheet("background: #ffffff; border-radius: 2px;")
        self.indicator.hide()
        self.textTimer = QTimer(self)
        self.textTimer.setSingleShot(True)
        self.textTimer.setInterval(40)
        self.textTimer.timeout.connect(self.refresh)

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # ---------- top bar
        top = QHBoxLayout()
        self.version = QComboBox()
        for p in sorted(PACKS.glob("*.pack"), key=lambda p: version_key(p.stem), reverse=True):
            self.version.addItem(p.stem, str(p))
        self.gpu = QRadioButton("GPU (CUDA)")
        self.cpu = QRadioButton("CPU")
        self.gpu.setChecked(True)
        if not (ENGINE / "seedwright-gpu").exists():
            self.gpu.setEnabled(False)
            self.cpu.setChecked(True)
        self.runBtn = QPushButton("▶  Search")
        self.runBtn.setDefault(True)
        self.runBtn.setStyleSheet("font-weight: 700; padding: 6px 18px;")
        self.stopBtn = QPushButton("■  Stop")
        self.stopBtn.setEnabled(False)
        self.mode = QComboBox()
        self.mode.addItems(["Find seeds", "Crack a seed"])
        self.mode.setToolTip("Find seeds: search for seeds matching blocks.\n"
                             "Crack a seed: work out a world's seed from structures you have seen in it.")
        top.addWidget(self.mode)
        top.addSpacing(16)
        top.addWidget(QLabel("Minecraft"))
        top.addWidget(self.version)
        top.addSpacing(16)
        top.addWidget(self.gpu)
        top.addWidget(self.cpu)
        top.addStretch()
        top.addWidget(self.stopBtn)
        top.addWidget(self.runBtn)
        outer.addLayout(top)

        vsplit = QSplitter(Qt.Vertical)
        outer.addWidget(vsplit, 1)
        split = QSplitter(Qt.Horizontal)
        self.stack = QStackedWidget()
        self.stack.addWidget(split)
        self.crack = CrackPanel(self)
        self.stack.addWidget(self.crack)
        vsplit.addWidget(self.stack)

        # ---------- palette
        self.palette = Palette(self)
        ps = QScrollArea()
        ps.setWidgetResizable(True)
        ps.setWidget(self.palette)
        ps.setMinimumWidth(170)
        split.addWidget(ps)

        # ---------- script + query text
        mid = QWidget()
        mv = QVBoxLayout(mid)
        mv.setContentsMargins(0, 0, 0, 0)
        hat = QFrame()
        hat.setObjectName("hat")
        hat.setStyleSheet("QFrame#hat { background: #2f6fd0; border: 2px solid #1f4f9a; border-radius: 12px; }"
                          "QFrame#hat > QLabel { color: white; font-weight: 700; font-size: 11pt; background: transparent; }"
                          "QFrame#mouth { background: rgba(0,0,0,0.30); border-radius: 8px; }"
                          "QLabel#empty { color: rgba(255,255,255,0.6); background: transparent; }")
        hv = QVBoxLayout(hat)
        self.hatLabel = QLabel()
        hv.addWidget(self.hatLabel)
        self.root = Mouth(self, None)
        self.root.empty.setText("drag blocks here from the palette")
        hv.addWidget(self.root)
        holder = QWidget()
        hl = QVBoxLayout(holder)
        hl.addWidget(hat)
        hl.addStretch()
        self.script = QScrollArea()
        self.script.setWidgetResizable(True)
        self.script.setWidget(holder)
        mv.addWidget(self.script, 1)
        qrow = QHBoxLayout()
        qrow.addWidget(QLabel("Query text (same language as the CLI's --where):"))
        qrow.addStretch()
        self.applyBtn = QPushButton("Text → blocks")
        self.applyBtn.setToolTip("Parse the text (seedwright parse) and rebuild the blocks from it")
        self.applyBtn.setEnabled(False)
        qrow.addWidget(self.applyBtn)
        mv.addLayout(qrow)
        self.qtext = QPlainTextEdit()
        mono = QFont("monospace")
        mono.setStyleHint(QFont.Monospace)
        self.qtext.setFont(mono)
        self.qtext.setMaximumHeight(90)
        self.qtext.setPlaceholderText("e.g.  2 village_plains within 500 { 1 blacksmith }, biome cherry_grove within 1500")
        mv.addWidget(self.qtext)
        self.qerr = QLabel()
        self.qerr.setStyleSheet("color: #ff6b6b;")
        self.qerr.setWordWrap(True)
        self.qerr.hide()
        mv.addWidget(self.qerr)
        split.addWidget(mid)

        # ---------- right: options; bottom: results + log
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        og = QGroupBox("Search options")
        g = QGridLayout(og)
        self.maxRes = SliderEntry(1, 1000, 20)
        self.slack = SliderEntry(0, 2048, 256, step=16, suffix=" b")
        self.slack.setToolTip("Speed shortcut: seeds whose spawn is farther than this from 0,0 are skipped.\n"
                              "Checking given seeds is always exact.")
        self.origin = QCheckBox("Measure from 0,0 instead of spawn")
        self.allBuildings = QCheckBox("Every village-building match (much slower)")
        self.allBuildings.setToolTip("Without this, a fast estimate pre-filters villages and keeps ~85% of the real\n"
                                     "matches; every result shown is still checked exactly (26.3).")
        self.count = Log2Entry(16, 48, 32)
        self.count.setToolTip("How many low-48 seeds to scan (--count); each survivor expands to 65 536 full seeds")
        self.start = QLineEdit("0")
        self.start.setValidator(QRegularExpressionValidator(QRegularExpression(r"\d{0,15}")))
        self.seeds = QLineEdit()
        self.seeds.setPlaceholderText("optional: only check these seeds (comma-separated)")
        for r, (lab, w) in enumerate([("Max results", self.maxRes), ("Spawn slack", self.slack), ("Seeds to scan", self.count),
                                      ("Start at", self.start), ("Check seeds", self.seeds)]):
            g.addWidget(QLabel(lab), r, 0)
            g.addWidget(w, r, 1)
        g.addWidget(self.origin, 5, 0, 1, 2)
        g.addWidget(self.allBuildings, 6, 0, 1, 2)
        g.setColumnStretch(1, 1)
        rv.addWidget(og)
        rv.addStretch()
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([190, 900, 330])
        bottom = QSplitter(Qt.Horizontal)
        resW = QWidget()
        resL = QVBoxLayout(resW)
        resL.setContentsMargins(0, 0, 0, 0)
        logW = QWidget()
        logL = QVBoxLayout(logW)
        logL.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Seed", "Spawn", "Found"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.cellDoubleClicked.connect(self.copySeed)
        resL.addWidget(QLabel("Results (double-click a row to copy its seed; nested finds are in { })"))
        resL.addWidget(self.table)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setFont(mono)
        logL.addWidget(QLabel("Engine log"))
        logL.addWidget(self.log)
        bottom.addWidget(resW)
        bottom.addWidget(logW)
        bottom.setSizes([1000, 420])
        vsplit.addWidget(bottom)
        vsplit.setSizes([620, 300])

        # ---------- bottom: command + status
        self.cmd = QLineEdit()
        self.cmd.setReadOnly(True)
        self.cmd.setFont(mono)
        copyCmd = QPushButton("Copy")
        copyCmd.clicked.connect(lambda: QGuiApplication.clipboard().setText(self.cmd.text()))
        clr = QPushButton("Clear results")
        clr.clicked.connect(lambda: (self.table.setRowCount(0), self.log.clear()))
        cl = QHBoxLayout()
        cl.addWidget(QLabel("Command"))
        cl.addWidget(self.cmd, 1)
        cl.addWidget(copyCmd)
        cl.addWidget(clr)
        outer.addLayout(cl)
        self.status = QLabel("idle")
        outer.addWidget(self.status)

        self.runBtn.clicked.connect(self.run)
        self.stopBtn.clicked.connect(self.stop)
        self.applyBtn.clicked.connect(self.applyText)
        self.qtext.textChanged.connect(self._textEdited)
        self.version.currentIndexChanged.connect(self.loadNames)
        for w in (self.gpu, self.cpu, self.origin, self.allBuildings):
            w.toggled.connect(self.queryChanged)
        self.mode.currentIndexChanged.connect(self.modeChanged)
        for w in (self.maxRes, self.slack):
            w.valueChanged.connect(self.queryChanged)
        self.count.valueChanged.connect(self.queryChanged)
        self.start.textChanged.connect(self.queryChanged)
        self.seeds.textChanged.connect(self.queryChanged)

        self.restore()

    # ---------------------------------------------------------------- blocks
    def allBlocks(self, mouth=None):
        out = []
        for b in (mouth or self.root).blocks():
            out.append(b)
            if b.mouth:
                out += self.allBlocks(b.mouth)
        return out

    def blockCount(self):
        return len(self.allBlocks())

    def blockById(self, i):
        for b in self.allBlocks():
            if b.id == i:
                return b
        return None

    def makeBlock(self, state):
        b = Block(self, state.get("kind", "structure"))
        if b.kind in ("structure", "biome"):
            b.fillNames(self.names)
        b.setState(state)
        return b

    def addBlock(self, state):
        if self.blockCount() >= MAX_BLOCKS:
            self.setStatus(f"at most {MAX_BLOCKS} blocks")
            return None
        b = self.makeBlock(state)
        self.root.insert(b)
        self.queryChanged()
        QTimer.singleShot(50, lambda: self.script.ensureWidgetVisible(b))
        return b

    def removeBlock(self, b):
        m = b.parentMouth()
        if m:
            m.take(b)
        b.hide()
        b.deleteLater()
        self.queryChanged()

    def clearBlocks(self):
        for b in self.root.blocks():
            self.root.take(b)
            b.hide()
            b.deleteLater()

    def showIndicator(self, mouth, index):
        if self.indicator.parentWidget() is not mouth or self.indicator.property("idx") != index:
            old = self.indicator.parentWidget()
            if isinstance(old, Mouth):
                old.lay.removeWidget(self.indicator)
            bl = mouth.blocks()
            pos = mouth.lay.indexOf(bl[index]) if index < len(bl) else mouth.lay.count() - 1
            mouth.lay.insertWidget(pos, self.indicator)
            self.indicator.setProperty("idx", index)
            self.indicator.show()

    def hideIndicator(self):
        old = self.indicator.parentWidget()
        if isinstance(old, Mouth):
            old.lay.removeWidget(self.indicator)
        self.indicator.hide()
        self.indicator.setParent(None)
        self.indicator.setProperty("idx", -1)

    # ---------------------------------------------------------------- names
    def loadNames(self):
        exe = engine_exe()
        names = {"sets": [], "structs": {}, "biomes": [], "buildings": {}}
        if exe:
            out = subprocess.run([str(exe), self.version.currentData(), "list"], capture_output=True, text=True, timeout=60).stdout
            for line in out.splitlines():
                p = line.split()
                if p[:1] == ["set"]:
                    names["sets"].append(p[1])
                    names["structs"][p[1]] = []
                elif p[:1] == ["structure"]:
                    names["structs"][p[1]].append(p[2])
                elif p[:1] == ["biome"]:
                    names["biomes"].append(p[1])
                elif p[:1] == ["building"]:
                    names["buildings"].setdefault(p[1], []).append(p[2])
        if not names["sets"]:
            QMessageBox.warning(self, "seedwright", "The engine gave no structure names. Rebuild it: cd engine && make")
        self.names = names
        self.palette.items["building"].setVisible(bool(names["buildings"]))
        for b in self.allBlocks():
            b.fillNames(names)
        self.crack.refreshNames()
        self.queryChanged()

    def structuresFor(self, name):
        """Structures a structure block's name can mean (set, exact, or * pattern)."""
        n = self.names
        if not name:
            return []
        if "*" in name:
            return [st for s in n["sets"] for st in n["structs"][s] if fnmatch.fnmatchcase(st, name)]
        for s in n["sets"]:
            if s == name or s == name + "s":
                return list(n["structs"][s])
        return [name]

    # ---------------------------------------------------------------- query
    def queryChanged(self, *_):
        if not self.loading:
            self.blocksChanged = True  # blocks win over unapplied typed text
            self.textTimer.start()

    def queryString(self):
        return ", ".join(b.query() for b in self.root.blocks())

    def refresh(self):
        # building pickers follow the village around them; mark blocks that can't work where they are
        for b in self.allBlocks():
            bad = ""
            if b.kind == "building":
                enc = b.enclosingStructure()
                opts = []
                if enc is not None:
                    for st in self.structuresFor(enc.targetValue()):
                        for x in self.names["buildings"].get(st, []):
                            if x not in opts:
                                opts.append(x)
                pending = getattr(b, "_pendingName", None)
                if sorted(opts) != sorted(getattr(b, "_opts", [])):
                    b._opts = opts
                    b.fillBuildings(sorted(opts))
                    if pending:
                        b.setTarget(pending)
                b._pendingName = None
                b.approx.setVisible(not self.exactBuildings())
                if enc is None:
                    bad = "Put this inside a village block (a structure block for villages)."
                elif not opts:
                    bad = f"'{enc.targetValue()}' has no building layouts; use a village."
            elif b.mouth is not None and b.kind in ("all", "any", "not") and not b.mouth.blocks():
                bad = "Empty group: drop blocks inside it."
            b.warn.setText(bad)
            b.warn.setVisible(bool(bad))
            b.restyle(bool(bad))
            if hasattr(b, "ofLabel") and b.kind != "spawn":
                parent = b.parentBlock()
                enclosing = parent
                while enclosing is not None and enclosing.kind not in ("structure", "stronghold"):
                    enclosing = enclosing.parentBlock()
                b.ofLabel.setText("of it" if enclosing is not None else ("of 0,0" if self.origin.isChecked() else "of spawn"))
        self.hatLabel.setText("🔍  Find seeds where, measured from " + ("0,0" if self.origin.isChecked() else "the world spawn") + ":")
        q = self.queryString()
        self.qtext.blockSignals(True)
        typed = self.applyBtn.isEnabled() and not getattr(self, "blocksChanged", True)
        if self.qtext.toPlainText() != q and not typed:
            self.qtext.setPlainText(q)
            self.qerr.hide()  # the text now comes from the blocks again
            self.applyBtn.setEnabled(False)
        self.qtext.blockSignals(False)
        self.blocksChanged = False
        self.updateCmd()

    def _textEdited(self):
        self.applyBtn.setEnabled(self.qtext.toPlainText().strip() != self.queryString())

    def parseText(self, text):
        exe = engine_exe()
        r = subprocess.run([str(exe), self.version.currentData(), "parse", text], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise ValueError(r.stderr.strip() or "parse failed")
        return json.loads(r.stdout)

    def applyText(self):
        text = self.qtext.toPlainText().strip()
        try:
            tree = self.parseText(text) if text else {"children": []}
        except ValueError as e:
            self.qerr.setText(str(e))
            self.qerr.show()
            return
        self.setTree(tree.get("children", []))
        self.blocksChanged = True
        self.refresh()

    def setTree(self, children):
        self.loading = True
        try:
            self.clearBlocks()
            for c in children:
                self.root.insert(self.makeBlock(c))
        finally:
            self.loading = False
        self.refresh()

    def exactBuildings(self):
        return self.version.currentText() == "26.3"

    def usedKinds(self):
        out, todo = set(), list(self.root.blocks())
        while todo:
            b = todo.pop()
            out.add(b.kind)
            if b.mouth is not None:
                todo += b.mouth.blocks()
        return out

    def modeChanged(self, *_):
        c = self.cracking()
        self.stack.setCurrentIndex(1 if c else 0)
        self.runBtn.setText("▶  Crack" if c else "▶  Search")
        for w in (self.gpu, self.cpu):
            w.setVisible(not c)
        if c and self.crack.table.rowCount() == 0:
            self.crack.addClue()
        self.queryChanged()

    def cracking(self):
        return self.mode.currentIndex() == 1

    def args(self):
        if self.cracking():
            return [self.version.currentData(), "crack"] + self.crack.args() + ["--max", str(self.maxRes.value())]
        a = [self.version.currentData(), "search", "--where", self.queryString()]
        if self.cpu.isChecked():
            a.append("--cpu")
        if self.origin.isChecked():
            a.append("--origin")
        if self.slack.value() != 256:
            a += ["--slack", str(self.slack.value())]
        if self.allBuildings.isChecked() and "building" in self.usedKinds():
            a.append("--all-buildings")
        a += ["--max", str(self.maxRes.value())]
        seeds = re.sub(r"\s+", "", self.seeds.text()).strip(",")
        if seeds:
            a += ["--seeds", seeds]
        else:
            a += ["--count", str(self.count.value())]
            if self.start.text() not in ("", "0"):
                a += ["--start", self.start.text()]
        return a

    def binary(self):
        if self.cracking():
            return ENGINE / "seedwright-cpu"  # the crack is CPU-only (seconds for enough clues)
        return ENGINE / ("seedwright-gpu" if self.gpu.isChecked() else "seedwright-cpu")

    def updateCmd(self, *_):
        a = self.args()
        a[0] = str(Path(a[0]).relative_to(ROOT))
        self.cmd.setText(shlex.join([str(self.binary().relative_to(ROOT))] + a))

    def setStatus(self, s):
        self.status.setText(s)

    # ---------------------------------------------------------------- run
    def run(self):
        if self.proc:
            return
        if self.cracking():
            if not self.crack.clues():
                QMessageBox.information(self, "seedwright", "Add the structures you have found (+ Add structure).")
                return
            self.start_proc("cracking…")
            return
        if self.applyBtn.isEnabled():
            self.applyText()
            if self.qerr.isVisible():
                return
        if not self.root.blocks():
            QMessageBox.information(self, "seedwright", "Drag at least one block into the script.")
            return
        bad = [s for s in re.split(r"[,\s]+", self.seeds.text().strip()) if s and not re.fullmatch(r"-?\d+", s)]
        if bad:
            QMessageBox.warning(self, "seedwright", f"Not a seed: {bad[0]}")
            return
        try:
            self.parseText(self.queryString())
        except ValueError as e:
            self.qerr.setText(str(e))
            self.qerr.show()
            self.setStatus("query error (see below the query text)")
            return
        self.start_proc("searching…")

    def start_proc(self, what):
        self.saveState()
        self.table.setRowCount(0)
        self.log.appendPlainText("$ " + self.cmd.text())
        self.errbuf = self.outbuf = ""
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(ROOT))
        self.proc.readyReadStandardOutput.connect(self.onOut)
        self.proc.readyReadStandardError.connect(self.onErr)
        self.proc.finished.connect(self.onDone)
        self.proc.errorOccurred.connect(lambda e: self.log.appendPlainText(f"process error: {e}"))
        self.proc.start(str(self.binary()), self.args())
        self.runBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        self.setStatus(what)

    def stop(self):
        if self.proc:
            self.proc.terminate()
            if not self.proc.waitForFinished(2000):
                self.proc.kill()

    def onOut(self):
        self.outbuf += bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        *lines, self.outbuf = self.outbuf.split("\n")
        for line in lines:
            self.addResult(line)

    def addResult(self, line):
        line = line.strip()
        if not line:
            return
        m = re.match(r"(-?\d+)\s+spawn\s+(-?\d+)\s+(-?\d+)\s*(.*)", line)
        if m:
            seed, sx, sz, rest = m.groups()
            ax, az = (0, 0) if self.origin.isChecked() else (int(sx), int(sz))
            row = [seed, f"{sx}, {sz}", self.formatFound(rest, ax, az)]
        elif re.match(r"-?\d+\s+no match", line):
            row = [line.split()[0], "", "no match"]
        elif re.fullmatch(r"-?\d+", line):  # crack: a world seed that fits every clue
            row = [line, "", f"fits all {len(self.crack.clues())} structures" + (" and the spawn" if self.crack.useSpawn.isChecked() else "")]
        else:
            self.log.appendPlainText(line)
            return
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, v in enumerate(row):
            it = QTableWidgetItem(v)
            if c == 0:
                it.setFont(QFont("monospace"))
            self.table.setItem(r, c, it)
        self.table.resizeRowToContents(r)
        self.setStatus(f"searching… {r + 1} result(s)")

    @staticmethod
    def formatFound(rest, ax=None, az=None):
        """' | village 1 2 | >weaponsmith 3 4' -> 'village (1, 2) { weaponsmith (3, 4) }' (depth = number of '>').
        With an anchor, top-level finds also say how far they are from it (the match may not be the nearest one)."""
        out, depth = "", 0
        first = True
        for p in rest.split("|"):
            p = p.strip()
            if not p:
                continue
            d = len(p) - len(p.lstrip(">"))
            f = p.lstrip(">").split()
            if len(f) != 3:
                continue
            item = f"{f[0]} ({f[1]}, {f[2]})"
            if d == 0 and ax is not None:
                item += f" {math.hypot(int(f[1]) - ax, int(f[2]) - az):,.0f} b away"
            if d > depth:
                out += " { " * (d - depth) + item
            else:
                out += " }" * (depth - d) + ("" if first else ",  ") + item
            depth, first = d, False
        return out + " }" * depth

    def onErr(self):
        self.errbuf += bytes(self.proc.readAllStandardError()).decode(errors="replace")
        *lines, self.errbuf = self.errbuf.split("\n")
        for line in lines:
            line = line.split("\r")[-1].strip()
            if line:
                self.log.appendPlainText(line)
        prog = [p for p in self.errbuf.split("\r") if p.strip()]
        if prog:
            self.setStatus(f"{self.table.rowCount()} result(s)  —  " + prog[-1].strip())
            self.errbuf = self.errbuf.split("\r")[-1]

    def onDone(self, code, _status):
        if self.outbuf:
            self.addResult(self.outbuf)
            self.outbuf = ""
        rest = self.errbuf.split("\r")[-1].strip()
        if rest:
            self.log.appendPlainText(rest)
        self.log.appendPlainText(f"[exit {code}]")
        self.setStatus(f"done: {self.table.rowCount()} result(s)" + ("" if code == 0 else f"  (exit {code}, see log)"))
        self.proc = None
        self.runBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)

    def copySeed(self, row, _col):
        QGuiApplication.clipboard().setText(self.table.item(row, 0).text())
        self.setStatus(f"copied {self.table.item(row, 0).text()}")

    # ---------------------------------------------------------------- persistence
    def saveState(self):
        s = {
            "version": self.version.currentText(), "cpu": self.cpu.isChecked(), "origin": self.origin.isChecked(),
            "allBuildings": self.allBuildings.isChecked(),
            "mode": self.mode.currentIndex(), "crack": self.crack.state(), "slack": self.slack.value(), "max": self.maxRes.value(), "count": self.count.text(),
            "start": self.start.text(), "seeds": self.seeds.text(), "blocks": [b.state() for b in self.root.blocks()],
        }
        self.settings.setValue("state2", json.dumps(s))
        self.settings.setValue("geometry2", self.saveGeometry())

    def restore(self):
        try:
            s = json.loads(self.settings.value("state2", "") or "{}")
        except ValueError:
            s = {}
        self.loading = True
        if s.get("version") and self.version.findText(s["version"]) >= 0:
            self.version.setCurrentText(s["version"])
        self.loading = False
        self.loadNames()
        if s.get("cpu") or not self.gpu.isEnabled():
            self.cpu.setChecked(True)
        self.origin.setChecked(bool(s.get("origin")))
        self.allBuildings.setChecked(bool(s.get("allBuildings")))
        self.crack.setState(s.get("crack", {}))
        self.mode.setCurrentIndex(s.get("mode", 0))
        self.slack.setValue(s.get("slack", 256))
        self.maxRes.setValue(s.get("max", 20))
        self.count.setText(s.get("count", "2^32"))
        self.start.setText(s.get("start", "0"))
        self.seeds.setText(s.get("seeds", ""))
        blocks = s.get("blocks")
        if blocks is None:
            blocks = [
                {"kind": "structure", "name": "village_plains", "count": 1, "radius": 500,
                 "children": [{"kind": "building", "name": "weaponsmith", "count": 1}]},
                {"kind": "biome", "name": "cherry_grove", "radius": 1500, "y": 64},
            ]
        self.setTree(blocks)
        g = self.settings.value("geometry2")
        if g:
            self.restoreGeometry(g)
        else:
            self.resize(1500, 900)

    def closeEvent(self, e):
        self.saveState()
        self.stop()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("seedwright")
    app.setDesktopFileName("seedwright-gui")
    w = Main()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

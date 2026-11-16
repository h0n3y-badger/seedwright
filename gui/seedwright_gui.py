#!/usr/bin/env python3
"""seedwright GUI: a front end for `seedwright-{gpu,cpu} <pack> search`.

Every control maps to one CLI flag; the exact command line is shown (and copyable) before it runs.
Run from anywhere: python3 gui/seedwright_gui.py
"""
import json
import re
import shlex
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QRegularExpression, QSettings, Qt, Signal
from PySide6.QtGui import QFont, QGuiApplication, QRegularExpressionValidator
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QRadioButton,
    QScrollArea, QSlider, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

ROOT = Path(__file__).resolve().parent.parent
PACKS = ROOT / "packs"
ENGINE = ROOT / "engine"
MAX_CONDS = 12  # query.h


def version_key(name):
    return [int(p) for p in re.findall(r"\d+", name)]


class SliderEntry(QWidget):
    """Slider plus an editable spin box, kept in sync. The entry accepts any value in range,
    the slider snaps to `step`."""
    valueChanged = Signal(int)

    def __init__(self, lo, hi, value, step=1, suffix=""):
        super().__init__()
        self.step = step
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(lo // step, hi // step)
        self.spin = QSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setSuffix(suffix)
        self.spin.setMinimumWidth(92)  # keyboard tracking stays on: the slider follows every keystroke
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
        self.spin.setValue(v)


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
        self.edit.setMinimumWidth(150)
        self.edit.setToolTip("Integer, or 2^N")
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.edit)
        self.slider.valueChanged.connect(lambda e: (self.edit.setText(f"2^{e}"), self.valueChanged.emit()))
        self.edit.textEdited.connect(lambda _: self._edited(live=True))  # follow typing
        self.edit.editingFinished.connect(self._edited)                  # on Enter/blur, reset invalid text
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
            if not live:  # half-typed text is fine while typing
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


KINDS = ["Structure", "Biome", "Stronghold", "Spawn within"]


class CondRow(QFrame):
    changed = Signal()
    removed = Signal(object)

    def __init__(self, names):
        super().__init__()
        self.names = names
        self.setFrameShape(QFrame.StyledPanel)
        g = QGridLayout(self)
        g.setContentsMargins(8, 6, 8, 6)
        self.kind = QComboBox()
        self.kind.addItems(KINDS)
        self.target = QComboBox()
        self.target.setEditable(True)
        self.target.setInsertPolicy(QComboBox.NoInsert)
        self.target.setMinimumWidth(260)
        self.target.completer().setFilterMode(Qt.MatchContains)
        self.target.completer().setCompletionMode(self.target.completer().CompletionMode.PopupCompletion)
        rm = QPushButton("✕")
        rm.setFixedWidth(32)
        rm.setToolTip("Remove this condition")
        self.count = SliderEntry(1, 16, 1)
        self.radius = SliderEntry(0, 20000, 500, step=16, suffix=" b")
        self.countLabel = QLabel("Count")
        g.addWidget(self.kind, 0, 0)
        g.addWidget(self.target, 0, 1, 1, 2)
        g.addWidget(rm, 0, 3)
        g.addWidget(self.countLabel, 1, 0)
        g.addWidget(self.count, 1, 1, 1, 3)
        g.addWidget(QLabel("Within"), 2, 0)
        g.addWidget(self.radius, 2, 1, 1, 3)
        g.setColumnStretch(1, 1)
        self.kind.currentIndexChanged.connect(self._kind)
        self.target.currentTextChanged.connect(lambda _: self.changed.emit())
        self.count.valueChanged.connect(lambda _: self.changed.emit())
        self.radius.valueChanged.connect(lambda _: self.changed.emit())
        rm.clicked.connect(lambda: self.removed.emit(self))
        self._kind()

    def setNames(self, names):
        self.names = names
        keep = self.target.currentText()
        self._fill()
        if keep:
            self._select(keep)

    def _fill(self):
        k = self.kind.currentText()
        self.target.blockSignals(True)
        self.target.clear()
        if k == "Structure":
            for s in self.names["sets"]:
                self.target.addItem(f"{s}  (any)", s)
                for st in self.names["structs"][s]:
                    if len(self.names["structs"][s]) > 1 or st not in s:
                        self.target.addItem(f"    {st}", "=" + st)
            # single-structure sets whose structure name is the set name minus a plural (monument etc.) stay set-only
        elif k == "Biome":
            for b in sorted(self.names["biomes"]):
                self.target.addItem(b, b)
        self.target.blockSignals(False)

    def _select(self, data):
        i = self.target.findData(data)
        if i < 0:
            i = self.target.findText(data.strip())
        if i >= 0:
            self.target.setCurrentIndex(i)

    def _kind(self):
        k = self.kind.currentText()
        self._fill()
        self.target.setVisible(k in ("Structure", "Biome"))
        self.count.setVisible(k in ("Structure", "Stronghold"))
        self.countLabel.setVisible(k in ("Structure", "Stronghold"))
        if k == "Stronghold" and self.radius.value() < 1500:
            self.radius.setValue(3000)  # the first ring starts ~1280 blocks out
        self.changed.emit()

    def targetValue(self):
        d = self.target.currentData()
        txt = self.target.currentText().strip()
        if d is not None and self.target.itemText(self.target.currentIndex()).strip() == txt:
            return d
        return txt.replace("  (any)", "")  # free text: CLI substring match

    def args(self):
        k, r = self.kind.currentText(), self.radius.value()
        if k == "Structure":
            return ["--need", f"{self.targetValue()}:{self.count.value()}@{r}"]
        if k == "Biome":
            return ["--biome", f"{self.targetValue()}@{r}"]
        if k == "Stronghold":
            return ["--stronghold", f"{self.count.value()}@{r}"]
        return ["--spawn-within", str(r)]

    def state(self):
        return {"kind": self.kind.currentText(), "target": self.targetValue(), "count": self.count.value(), "radius": self.radius.value()}

    def setState(self, s):
        self.kind.setCurrentText(s.get("kind", "Structure"))
        self._select(s.get("target", ""))
        self.count.setValue(s.get("count", 1))
        if "radius" in s:
            self.radius.setValue(s["radius"])


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("seedwright")
        self.settings = QSettings("seedwright", "gui")
        self.proc = None
        self.rows = []
        self.names = {"sets": [], "structs": {}, "biomes": []}
        self.errbuf = ""

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split, 1)

        # ---------- left: query ----------
        left = QWidget()
        lv = QVBoxLayout(left)
        top = QGridLayout()
        self.version = QComboBox()
        for p in sorted(PACKS.glob("*.pack"), key=lambda p: version_key(p.stem), reverse=True):
            self.version.addItem(p.stem, str(p))
        self.gpu = QRadioButton("GPU (CUDA)")
        self.cpu = QRadioButton("CPU")
        self.gpu.setChecked(True)
        if not (ENGINE / "seedwright-gpu").exists():
            self.gpu.setEnabled(False)
            self.cpu.setChecked(True)
        be = QHBoxLayout()
        be.addWidget(self.gpu)
        be.addWidget(self.cpu)
        be.addStretch()
        top.addWidget(QLabel("Version"), 0, 0)
        top.addWidget(self.version, 0, 1)
        top.addWidget(QLabel("Backend"), 1, 0)
        top.addLayout(be, 1, 1)
        top.setColumnStretch(1, 1)
        lv.addLayout(top)

        cg = QGroupBox("Conditions (all must hold)")
        cv = QVBoxLayout(cg)
        self.condBox = QVBoxLayout()
        self.condBox.addStretch()
        cw = QWidget()
        cw.setLayout(self.condBox)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setWidget(cw)
        sa.setFrameShape(QFrame.NoFrame)
        cv.addWidget(sa, 1)
        addRow = QHBoxLayout()
        for k in KINDS:
            b = QPushButton("+ " + k)
            b.clicked.connect(lambda _=False, k=k: self.addCond({"kind": k}))
            addRow.addWidget(b)
        self.addButtons = addRow
        cv.addLayout(addRow)
        lv.addWidget(cg, 1)

        og = QGroupBox("Options")
        og_l = QGridLayout(og)
        self.maxRes = SliderEntry(1, 1000, 20)
        self.slack = SliderEntry(0, 2048, 256, step=16, suffix=" b")
        self.slack.setToolTip("How far spawn may be from 0,0 when pre-filtering structures (--slack)")
        self.origin = QCheckBox("Measure distances from 0,0 instead of spawn (--origin)")
        self.count = Log2Entry(16, 48, 32)
        self.count.setToolTip("How many low-48 seeds to scan (--count); each survivor expands to 65 536 full seeds")
        self.start = QLineEdit("0")
        self.start.setValidator(QRegularExpressionValidator(QRegularExpression(r"\d{0,15}")))
        self.start.setToolTip("First low-48 seed (--start)")
        self.seeds = QLineEdit()
        self.seeds.setPlaceholderText("optional: check only these seeds, comma-separated (--seeds)")
        og_l.addWidget(QLabel("Max results"), 0, 0)
        og_l.addWidget(self.maxRes, 0, 1)
        og_l.addWidget(QLabel("Spawn slack"), 1, 0)
        og_l.addWidget(self.slack, 1, 1)
        og_l.addWidget(QLabel("Low-48 range"), 2, 0)
        og_l.addWidget(self.count, 2, 1)
        og_l.addWidget(QLabel("Start at"), 3, 0)
        og_l.addWidget(self.start, 3, 1)
        og_l.addWidget(self.origin, 4, 0, 1, 2)
        og_l.addWidget(QLabel("Seeds"), 5, 0)
        og_l.addWidget(self.seeds, 5, 1)
        og_l.setColumnStretch(1, 1)
        lv.addWidget(og)
        split.addWidget(left)

        # ---------- right: results ----------
        right = QWidget()
        rv = QVBoxLayout(right)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Seed", "Spawn", "Matches"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.cellDoubleClicked.connect(self.copySeed)
        rv.addWidget(QLabel("Results (double-click a row to copy its seed)"))
        rv.addWidget(self.table, 3)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        mono = QFont("monospace")
        mono.setStyleHint(QFont.Monospace)
        self.log.setFont(mono)
        rv.addWidget(QLabel("Engine log"))
        rv.addWidget(self.log, 1)
        split.addWidget(right)
        split.setSizes([560, 640])

        # ---------- bottom: command + run ----------
        self.cmd = QLineEdit()
        self.cmd.setReadOnly(True)
        self.cmd.setFont(mono)
        copyCmd = QPushButton("Copy")
        copyCmd.clicked.connect(lambda: QGuiApplication.clipboard().setText(self.cmd.text()))
        cl = QHBoxLayout()
        cl.addWidget(QLabel("Command"))
        cl.addWidget(self.cmd, 1)
        cl.addWidget(copyCmd)
        outer.addLayout(cl)
        bl = QHBoxLayout()
        self.status = QLabel("idle")
        self.runBtn = QPushButton("Search")
        self.runBtn.setDefault(True)
        self.stopBtn = QPushButton("Stop")
        self.stopBtn.setEnabled(False)
        clr = QPushButton("Clear results")
        clr.clicked.connect(lambda: (self.table.setRowCount(0), self.log.clear()))
        bl.addWidget(self.status, 1)
        bl.addWidget(clr)
        bl.addWidget(self.stopBtn)
        bl.addWidget(self.runBtn)
        outer.addLayout(bl)

        self.runBtn.clicked.connect(self.run)
        self.stopBtn.clicked.connect(self.stop)
        self.version.currentIndexChanged.connect(self.loadNames)
        for w in (self.gpu, self.cpu, self.origin):
            w.toggled.connect(self.updateCmd)
        for w in (self.maxRes, self.slack):
            w.valueChanged.connect(self.updateCmd)
        self.count.valueChanged.connect(self.updateCmd)
        self.start.textChanged.connect(self.updateCmd)
        self.seeds.textChanged.connect(self.updateCmd)

        self.restore()
        self.updateCmd()

    # ---------- names / conditions ----------
    def binary(self):
        return ENGINE / ("seedwright-gpu" if self.gpu.isChecked() else "seedwright-cpu")

    def loadNames(self):
        pack = self.version.currentData()
        exe = ENGINE / "seedwright-cpu"
        if not exe.exists():
            exe = ENGINE / "seedwright-gpu"
        p = QProcess()
        p.start(str(exe), [pack, "list"])
        p.waitForFinished(30000)
        names = {"sets": [], "structs": {}, "biomes": []}
        for line in bytes(p.readAllStandardOutput()).decode().splitlines():
            parts = line.split()
            if parts[:1] == ["set"]:
                names["sets"].append(parts[1])
                names["structs"][parts[1]] = []
            elif parts[:1] == ["structure"]:
                names["structs"][parts[1]].append(parts[2])
            elif parts[:1] == ["biome"]:
                names["biomes"].append(parts[1])
        if not names["sets"]:
            QMessageBox.warning(self, "seedwright", f"`{exe.name} {Path(pack).name} list` gave no names.\n"
                                "Rebuild the engine (cd engine && make).")
        self.names = names
        for r in self.rows:
            r.setNames(names)
        self.updateCmd()

    def addCond(self, state):
        if len(self.rows) >= MAX_CONDS:
            return
        r = CondRow(self.names)
        r.setState(state)
        r.changed.connect(self.updateCmd)
        r.removed.connect(self.removeCond)
        self.condBox.insertWidget(self.condBox.count() - 1, r)
        self.rows.append(r)
        self.updateCmd()

    def removeCond(self, r):
        self.rows.remove(r)
        r.hide()
        self.condBox.removeWidget(r)
        r.deleteLater()
        self.updateCmd()

    # ---------- command ----------
    def args(self):
        a = [self.version.currentData(), "search"]
        for r in self.rows:
            a += r.args()
        if self.cpu.isChecked():
            a.append("--cpu")
        if self.origin.isChecked():
            a.append("--origin")
        if self.slack.value() != 256:
            a += ["--slack", str(self.slack.value())]
        a += ["--max", str(self.maxRes.value())]
        seeds = re.sub(r"[\s]+", "", self.seeds.text()).strip(",")
        if seeds:
            a += ["--seeds", seeds]
        else:
            a += ["--count", str(self.count.value())]
            if self.start.text() not in ("", "0"):
                a += ["--start", self.start.text()]
        return a

    def updateCmd(self, *_):
        if not hasattr(self, "cmd"):
            return
        a = self.args()
        a[0] = str(Path(a[0]).relative_to(ROOT))
        self.cmd.setText(shlex.join([str(self.binary().relative_to(ROOT))] + a))
        for i in range(self.addButtons.count()):
            self.addButtons.itemAt(i).widget().setEnabled(len(self.rows) < MAX_CONDS)

    # ---------- run ----------
    def run(self):
        if self.proc:
            return
        if not self.rows:
            QMessageBox.information(self, "seedwright", "Add at least one condition.")
            return
        bad = [s for s in re.split(r"[,\s]+", self.seeds.text().strip()) if s and not re.fullmatch(r"-?\d+", s)]
        if bad:
            QMessageBox.warning(self, "seedwright", f"Not a seed: {bad[0]}")
            return
        self.saveState()
        self.table.setRowCount(0)
        self.log.appendPlainText("$ " + self.cmd.text())
        self.errbuf = ""
        self.outbuf = ""
        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(ROOT))
        self.proc.readyReadStandardOutput.connect(self.onOut)
        self.proc.readyReadStandardError.connect(self.onErr)
        self.proc.finished.connect(self.onDone)
        self.proc.errorOccurred.connect(lambda e: self.log.appendPlainText(f"process error: {e}"))
        self.proc.start(str(self.binary()), self.args())
        self.runBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        self.status.setText("searching…")

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
            matches = "\n".join(f"{p.split()[0]} ({p.split()[1]}, {p.split()[2]})"
                                 for p in rest.split("|") if len(p.split()) == 3)
            row = [seed, f"{sx}, {sz}", matches]
        elif re.match(r"-?\d+\s+no match", line):
            row = [line.split()[0], "", "no match"]
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
        self.status.setText(f"searching… {r + 1} result(s)")

    def onErr(self):
        self.errbuf += bytes(self.proc.readAllStandardError()).decode(errors="replace")
        # stage-B progress uses \r: show it in the status line, keep finished lines in the log
        *lines, self.errbuf = re.split(r"\n", self.errbuf)
        for line in lines:
            line = line.split("\r")[-1].strip()
            if line:
                self.log.appendPlainText(line)
        prog = self.errbuf.split("\r")
        if len([p for p in prog if p.strip()]):
            self.status.setText(f"{self.table.rowCount()} result(s)  —  " + [p for p in prog if p.strip()][-1].strip())
            self.errbuf = prog[-1]

    def onDone(self, code, _status):
        if self.outbuf:
            self.addResult(self.outbuf)
            self.outbuf = ""
        rest = self.errbuf.split("\r")[-1].strip()
        if rest:
            self.log.appendPlainText(rest)
        self.log.appendPlainText(f"[exit {code}]")
        self.status.setText(f"done: {self.table.rowCount()} result(s)" + ("" if code == 0 else f"  (exit {code})"))
        self.proc = None
        self.runBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)

    def copySeed(self, row, _col):
        QGuiApplication.clipboard().setText(self.table.item(row, 0).text())
        self.status.setText(f"copied {self.table.item(row, 0).text()}")

    # ---------- persistence ----------
    def saveState(self):
        s = {
            "version": self.version.currentText(), "cpu": self.cpu.isChecked(), "origin": self.origin.isChecked(),
            "slack": self.slack.value(), "max": self.maxRes.value(), "count": self.count.text(),
            "start": self.start.text(), "seeds": self.seeds.text(), "conds": [r.state() for r in self.rows],
        }
        self.settings.setValue("state", json.dumps(s))
        self.settings.setValue("geometry", self.saveGeometry())

    def restore(self):
        try:
            s = json.loads(self.settings.value("state", "") or "{}")
        except ValueError:
            s = {}
        if s.get("version") and self.version.findText(s["version"]) >= 0:
            self.version.setCurrentText(s["version"])
        self.loadNames()
        if s.get("cpu") or not self.gpu.isEnabled():
            self.cpu.setChecked(True)
        self.origin.setChecked(bool(s.get("origin")))
        self.slack.setValue(s.get("slack", 256))
        self.maxRes.setValue(s.get("max", 20))
        self.count.setText(s.get("count", "2^32"))
        self.start.setText(s.get("start", "0"))
        self.seeds.setText(s.get("seeds", ""))
        conds = s.get("conds") or [
            {"kind": "Structure", "target": "villages", "count": 1, "radius": 300},
            {"kind": "Biome", "target": "cherry_grove", "radius": 1500},
        ]
        for c in conds:
            self.addCond(c)
        g = self.settings.value("geometry")
        if g:
            self.restoreGeometry(g)
        else:
            self.resize(1250, 780)

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

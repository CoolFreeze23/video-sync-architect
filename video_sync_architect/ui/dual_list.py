"""
Dual-list file selection widget (subsync-style).

- Left list  = Primary Reference
- Right list = Secondary Target
- Items are paired by row index (row 0 left <-> row 0 right, etc.)
- Vertical scrolling is synchronized between the two lists.
- Both lists accept drag-and-drop from the OS file explorer.
- Both lists support internal drag-reordering.
- Auto-sort button realigns the right list to maximize filename similarity
  with the left list, prompting on low-confidence matches.
"""

import os
import difflib
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QDragEnterEvent, QDropEvent, QDragMoveEvent, QIcon
from PyQt6.QtWidgets import (
    QWidget, QListWidget, QListWidgetItem, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QAbstractItemView, QFileDialog, QMessageBox, QFrame,
    QSizePolicy, QDialog, QDialogButtonBox, QCheckBox,
)


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".mxf", ".ts", ".webm", ".flv", ".wmv"}

# Auto-sort thresholds.
HIGH_CONFIDENCE = 0.65   # Above this: accept silently.
LOW_CONFIDENCE = 0.35    # Below this: prompt the user.
# Between low and high: also prompt (medium confidence).


# ---------------------------------------------------------------------------
# Custom QListWidget that accepts external file drops and supports reorder.
# ---------------------------------------------------------------------------

class FileListWidget(QListWidget):
    files_added = pyqtSignal(list)            # list[str] of newly added paths
    order_changed = pyqtSignal()              # emitted after internal reorder

    def __init__(self, label: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._label_text = label

        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setUniformItemSizes(True)
        self.setMinimumHeight(220)
        self.setStyleSheet(
            "QListWidget { background: #2b2b3b; color: #e0e0e0; border: 1px solid #444; "
            "border-radius: 6px; padding: 4px; font-size: 12px; }"
            "QListWidget::item { padding: 6px 8px; border-radius: 3px; }"
            "QListWidget::item:alternate { background: #2f2f40; }"
            "QListWidget::item:selected { background: #3a7bd5; color: white; }"
            "QListWidget::item:hover { background: #3b3b50; }"
        )

    # --- Public API ---------------------------------------------------------

    def add_file(self, filepath: str) -> bool:
        if not filepath:
            return False
        norm = os.path.normpath(filepath)
        if not os.path.isfile(norm):
            return False
        ext = os.path.splitext(norm)[1].lower()
        if ext not in VIDEO_EXTENSIONS:
            return False
        for i in range(self.count()):
            if self.item(i).data(Qt.ItemDataRole.UserRole) == norm:
                return False  # duplicate
        item = QListWidgetItem(os.path.basename(norm))
        item.setData(Qt.ItemDataRole.UserRole, norm)
        item.setToolTip(norm)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsDragEnabled
                      | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
        self.addItem(item)
        return True

    def add_files(self, paths: list[str]) -> int:
        added = []
        for p in paths:
            if os.path.isdir(p):
                for entry in sorted(os.listdir(p)):
                    full = os.path.join(p, entry)
                    if self.add_file(full):
                        added.append(full)
            else:
                if self.add_file(p):
                    added.append(p)
        if added:
            self.files_added.emit(added)
        return len(added)

    def all_paths(self) -> list[str]:
        return [self.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.count())]

    def remove_selected(self):
        for item in self.selectedItems():
            self.takeItem(self.row(item))

    def clear_all(self):
        self.clear()

    def reorder_paths(self, new_order: list[str]):
        """Reorder the list to exactly match new_order (paths must already exist)."""
        existing = {self.item(i).data(Qt.ItemDataRole.UserRole): self.item(i)
                    for i in range(self.count())}
        # Detach all items, then re-add in the new order.
        for i in reversed(range(self.count())):
            self.takeItem(i)
        for path in new_order:
            if path in existing:
                item = existing[path]
                self.addItem(item)
        self.order_changed.emit()

    # --- Drag & drop overrides ---------------------------------------------

    def _has_external_urls(self, event) -> bool:
        md = event.mimeData()
        return md.hasUrls() and any(u.isLocalFile() for u in md.urls())

    def dragEnterEvent(self, event: QDragEnterEvent):
        if self._has_external_urls(event):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QDragMoveEvent):
        if self._has_external_urls(event):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event: QDropEvent):
        if self._has_external_urls(event):
            paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
            self.add_files(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
            self.order_changed.emit()


# ---------------------------------------------------------------------------
# Confirmation dialog for low-confidence pairings.
# ---------------------------------------------------------------------------

class PairConfirmDialog(QDialog):
    """Asks the user to confirm or reject a single suggested pair."""

    def __init__(self, primary_name: str, suggested_name: str,
                 similarity: float, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Confirm Suggested Pairing")
        self.setMinimumWidth(560)
        self._accepted = False
        self._skip = False

        layout = QVBoxLayout(self)

        intro = QLabel(
            f"<b>Low-confidence pairing detected ({similarity:.0%} similarity).</b><br>"
            "Please confirm or reject this pair."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #e0e0e0;")
        layout.addWidget(intro)

        primary_lbl = QLabel(f"<b>Primary:</b><br><span style='color:#9bd5ff;'>{primary_name}</span>")
        primary_lbl.setWordWrap(True)
        primary_lbl.setStyleSheet("padding: 8px; background: #2b2b3b; border-radius: 4px;")
        layout.addWidget(primary_lbl)

        target_lbl = QLabel(f"<b>Suggested Target:</b><br><span style='color:#ffd59b;'>{suggested_name}</span>")
        target_lbl.setWordWrap(True)
        target_lbl.setStyleSheet("padding: 8px; background: #2b2b3b; border-radius: 4px;")
        layout.addWidget(target_lbl)

        bb = QDialogButtonBox()
        accept_btn = bb.addButton("Accept Pair", QDialogButtonBox.ButtonRole.AcceptRole)
        reject_btn = bb.addButton("Reject (leave unmatched)", QDialogButtonBox.ButtonRole.RejectRole)
        accept_btn.clicked.connect(self._on_accept)
        reject_btn.clicked.connect(self._on_reject)
        layout.addWidget(bb)

    def _on_accept(self):
        self._accepted = True
        self.accept()

    def _on_reject(self):
        self._accepted = False
        self.reject()

    def was_accepted(self) -> bool:
        return self._accepted


# ---------------------------------------------------------------------------
# DualFileListWidget: two FileListWidgets side-by-side with toolbar.
# ---------------------------------------------------------------------------

class DualFileListWidget(QWidget):
    """
    The full subsync-style dual-list panel.

    Public API:
      pairs() -> list[tuple[str, str]]  paths for rows where both lists have items.
      counts() -> tuple[int, int]
    """

    pairs_changed = pyqtSignal()  # emitted after add/remove/reorder/auto-sort

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # --- Header labels for the two columns ---
        header_row = QHBoxLayout()
        header_row.setSpacing(10)

        primary_lbl = QLabel("◄  Primary Reference")
        primary_lbl.setStyleSheet(
            "color: #9bd5ff; font-weight: bold; font-size: 13px; padding: 4px;"
        )
        primary_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_row.addWidget(primary_lbl, stretch=1)

        target_lbl = QLabel("Secondary Target  ►")
        target_lbl.setStyleSheet(
            "color: #ffd59b; font-weight: bold; font-size: 13px; padding: 4px;"
        )
        target_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_row.addWidget(target_lbl, stretch=1)

        outer.addLayout(header_row)

        # --- The two list widgets ---
        lists_row = QHBoxLayout()
        lists_row.setSpacing(10)

        self.primary_list = FileListWidget("Primary Reference")
        self.target_list = FileListWidget("Secondary Target")

        lists_row.addWidget(self.primary_list, stretch=1)
        lists_row.addWidget(self.target_list, stretch=1)

        outer.addLayout(lists_row, stretch=1)

        # --- Toolbar / action buttons ---
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        self.btn_add_files_primary = self._make_btn("Add Files (Primary)", "#3a7bd5")
        self.btn_add_files_target = self._make_btn("Add Files (Target)", "#3a7bd5")
        self.btn_add_folder = self._make_btn("Add Folder...", "#5d6d7e")
        self.btn_remove = self._make_btn("Remove Selected", "#7d3c3c")
        self.btn_clear = self._make_btn("Clear All", "#7d3c3c")
        self.btn_autosort = self._make_btn("Auto-Sort", "#27ae60")

        toolbar.addWidget(self.btn_add_files_primary)
        toolbar.addWidget(self.btn_add_files_target)
        toolbar.addWidget(self.btn_add_folder)
        toolbar.addStretch(1)
        toolbar.addWidget(self.btn_remove)
        toolbar.addWidget(self.btn_clear)
        toolbar.addWidget(self.btn_autosort)

        outer.addLayout(toolbar)

        # --- Wire up scroll synchronization ---
        self._sync_guard = False
        self.primary_list.verticalScrollBar().valueChanged.connect(self._sync_scroll_from_primary)
        self.target_list.verticalScrollBar().valueChanged.connect(self._sync_scroll_from_target)

        # --- Wire up signals ---
        self.btn_add_files_primary.clicked.connect(self._add_files_primary)
        self.btn_add_files_target.clicked.connect(self._add_files_target)
        self.btn_add_folder.clicked.connect(self._add_folder)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear.clicked.connect(self._clear_all)
        self.btn_autosort.clicked.connect(self.auto_sort)

        for lst in (self.primary_list, self.target_list):
            lst.files_added.connect(lambda *_: self.pairs_changed.emit())
            lst.order_changed.connect(self.pairs_changed.emit)
            lst.model().rowsRemoved.connect(lambda *_: self.pairs_changed.emit())

    # --- Helpers ------------------------------------------------------------

    def _make_btn(self, text: str, color: str) -> QPushButton:
        b = QPushButton(text)
        b.setFixedHeight(32)
        b.setStyleSheet(
            f"QPushButton {{ background: {color}; color: white; border: none; "
            f"border-radius: 5px; padding: 0 14px; font-weight: bold; font-size: 12px; }}"
            f"QPushButton:hover {{ background: {color}; opacity: 0.9; }}"
            f"QPushButton:disabled {{ background: #555; color: #888; }}"
        )
        return b

    def _sync_scroll_from_primary(self, value: int):
        if self._sync_guard:
            return
        self._sync_guard = True
        try:
            self.target_list.verticalScrollBar().setValue(value)
        finally:
            self._sync_guard = False

    def _sync_scroll_from_target(self, value: int):
        if self._sync_guard:
            return
        self._sync_guard = True
        try:
            self.primary_list.verticalScrollBar().setValue(value)
        finally:
            self._sync_guard = False

    # --- Toolbar actions ----------------------------------------------------

    def _video_filter(self) -> str:
        exts = " ".join(f"*{e}" for e in sorted(VIDEO_EXTENSIONS))
        return f"Video Files ({exts});;All Files (*)"

    def _add_files_primary(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Primary Files", "", self._video_filter()
        )
        if paths:
            self.primary_list.add_files(paths)

    def _add_files_target(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Target Files", "", self._video_filter()
        )
        if paths:
            self.target_list.add_files(paths)

    def _add_folder(self):
        primary_dir = QFileDialog.getExistingDirectory(
            self, "Select PRIMARY folder (cancel to skip)"
        )
        if primary_dir:
            self.primary_list.add_files([primary_dir])

        target_dir = QFileDialog.getExistingDirectory(
            self, "Select TARGET folder (cancel to skip)"
        )
        if target_dir:
            self.target_list.add_files([target_dir])

        if primary_dir and target_dir and self.primary_list.count() and self.target_list.count():
            ans = QMessageBox.question(
                self, "Auto-Sort?",
                "Folder import complete. Run Auto-Sort now to align the two lists?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans == QMessageBox.StandardButton.Yes:
                self.auto_sort()

    def _remove_selected(self):
        self.primary_list.remove_selected()
        self.target_list.remove_selected()
        self.pairs_changed.emit()

    def _clear_all(self):
        self.primary_list.clear_all()
        self.target_list.clear_all()
        self.pairs_changed.emit()

    # --- Public API ---------------------------------------------------------

    def pairs(self) -> list[tuple[str, str]]:
        primary = self.primary_list.all_paths()
        target = self.target_list.all_paths()
        n = min(len(primary), len(target))
        return [(primary[i], target[i]) for i in range(n)]

    def counts(self) -> tuple[int, int]:
        return self.primary_list.count(), self.target_list.count()

    # --- Auto-sort ----------------------------------------------------------

    def auto_sort(self):
        """
        Realign the RIGHT (target) list so each row's target best matches
        the same row's primary by filename similarity.
        Halts to ask the user when confidence is low.
        """
        primary_paths = self.primary_list.all_paths()
        target_paths = self.target_list.all_paths()

        if not primary_paths or not target_paths:
            QMessageBox.information(
                self, "Auto-Sort",
                "Both lists must contain at least one file before auto-sorting."
            )
            return

        remaining = list(target_paths)
        new_order: list[Optional[str]] = []

        for primary in primary_paths:
            if not remaining:
                new_order.append(None)
                continue

            p_norm = _normalize_name(primary)
            scored = []
            for t in remaining:
                t_norm = _normalize_name(t)
                score = difflib.SequenceMatcher(None, p_norm, t_norm).ratio()
                scored.append((score, t))
            scored.sort(reverse=True, key=lambda x: x[0])

            best_score, best_target = scored[0]

            if best_score >= HIGH_CONFIDENCE:
                new_order.append(best_target)
                remaining.remove(best_target)
            else:
                # Low or medium confidence: prompt user.
                dlg = PairConfirmDialog(
                    os.path.basename(primary),
                    os.path.basename(best_target),
                    best_score,
                    parent=self,
                )
                dlg.exec()
                if dlg.was_accepted():
                    new_order.append(best_target)
                    remaining.remove(best_target)
                else:
                    new_order.append(None)

        # Append any remaining unmatched targets to the end so they
        # are not lost (user can manually drag them into place).
        final_targets = [t for t in new_order if t is not None] + remaining

        # Pad with empty placeholders if shorter than primary, so row
        # alignment is preserved visually. We do this by inserting a
        # blank "(missing)" item.
        self.target_list.clear_all()
        idx = 0
        for slot in new_order:
            if slot is None:
                placeholder = QListWidgetItem("⨯  (no match - drag a file here)")
                placeholder.setData(Qt.ItemDataRole.UserRole, "")
                placeholder.setForeground(Qt.GlobalColor.gray)
                placeholder.setFlags(
                    Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsDragEnabled
                )
                self.target_list.addItem(placeholder)
            else:
                self.target_list.add_file(slot)

        # Append leftovers below.
        for leftover in remaining:
            self.target_list.add_file(leftover)

        self.pairs_changed.emit()

        QMessageBox.information(
            self, "Auto-Sort Complete",
            f"Aligned {sum(1 for s in new_order if s is not None)} pair(s). "
            f"{len(remaining)} target file(s) left at the bottom unmatched."
        )


# ---------------------------------------------------------------------------
# Local copy of normalization (avoids importing from core to keep UI standalone).
# ---------------------------------------------------------------------------

def _normalize_name(filepath: str) -> str:
    name = os.path.splitext(os.path.basename(filepath))[0]
    for suffix in ("_primary", "_anchor", "_cam1", "_cam2", "_target", "_synced"):
        name = name.replace(suffix, "")
    name = name.lower().strip()
    # Collapse common separators and noise to improve matching.
    for ch in ("_", "-", ".", "  "):
        name = name.replace(ch, " ")
    return " ".join(name.split())

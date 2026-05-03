"""
Custom PyQt6 widgets: console log, styled progress bars, path selector.
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QTextEdit, QProgressBar, QSlider,
    QSpinBox, QGroupBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor, QColor


class PathSelector(QWidget):
    """A row with label, text field, and browse button for file or folder."""

    path_changed = pyqtSignal(str)

    def __init__(self, label: str, mode: str = "file", parent=None):
        super().__init__(parent)
        self._mode = mode
        self._filter = "Video Files (*.mp4 *.mkv *.mov *.avi *.mxf *.ts *.webm);;All Files (*)"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._label = QLabel(label)
        self._label.setFixedWidth(130)
        self._label.setStyleSheet("font-weight: bold; color: #e0e0e0;")
        layout.addWidget(self._label)

        self._edit = QLineEdit()
        self._edit.setPlaceholderText(
            "Select a folder..." if mode == "folder" else "Select a file..."
        )
        self._edit.setStyleSheet(
            "QLineEdit { background: #2b2b2b; color: #e0e0e0; border: 1px solid #555; "
            "border-radius: 4px; padding: 6px; font-size: 13px; }"
        )
        self._edit.textChanged.connect(self.path_changed.emit)
        layout.addWidget(self._edit, stretch=1)

        self._btn = QPushButton("Browse")
        self._btn.setFixedWidth(80)
        self._btn.setStyleSheet(
            "QPushButton { background: #3a7bd5; color: white; border: none; "
            "border-radius: 4px; padding: 6px 12px; font-weight: bold; }"
            "QPushButton:hover { background: #4a8be5; }"
            "QPushButton:pressed { background: #2a6bc5; }"
        )
        self._btn.clicked.connect(self._browse)
        layout.addWidget(self._btn)

    def _browse(self):
        if self._mode == "folder":
            path = QFileDialog.getExistingDirectory(self, "Select Folder")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", "", self._filter)
        if path:
            self._edit.setText(path)

    def path(self) -> str:
        return self._edit.text().strip()

    def set_path(self, path: str):
        self._edit.setText(path)


class ConsoleLog(QTextEdit):
    """Dark-themed read-only console log with auto-scroll."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 11))
        self.setStyleSheet(
            "QTextEdit { background: #1a1a2e; color: #00ff88; border: 1px solid #444; "
            "border-radius: 6px; padding: 8px; }"
        )
        self.setMinimumHeight(200)

    def append_log(self, message: str, color: str = "#00ff88"):
        if not message:
            return
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.setTextCursor(cursor)

        html = f'<span style="color:{color};">{message}</span><br>'
        self.insertHtml(html)
        self.ensureCursorVisible()

    def append_info(self, message: str):
        self.append_log(f"> {message}", "#00ff88")

    def append_warning(self, message: str):
        self.append_log(f"⚠ {message}", "#ffcc00")

    def append_error(self, message: str):
        self.append_log(f"✖ {message}", "#ff4444")

    def append_success(self, message: str):
        self.append_log(f"✔ {message}", "#44ff44")


class StyledProgressBar(QWidget):
    """Labeled progress bar with percentage display."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)

        self._label = QLabel(label)
        self._label.setFixedWidth(130)
        self._label.setStyleSheet("color: #c0c0c0; font-weight: bold;")
        layout.addWidget(self._label)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(22)
        self._bar.setStyleSheet(
            "QProgressBar { background: #2b2b2b; border: 1px solid #555; border-radius: 6px; }"
            "QProgressBar::chunk { background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:0, stop:0 #3a7bd5, stop:1 #00d2ff); border-radius: 5px; }"
        )
        layout.addWidget(self._bar, stretch=1)

        self._pct = QLabel("0%")
        self._pct.setFixedWidth(45)
        self._pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._pct.setStyleSheet("color: #e0e0e0; font-weight: bold;")
        layout.addWidget(self._pct)

    def set_progress(self, fraction: float):
        value = max(0, min(1000, int(fraction * 1000)))
        self._bar.setValue(value)
        self._pct.setText(f"{fraction * 100:.0f}%")

    def reset(self):
        self._bar.setValue(0)
        self._pct.setText("0%")


class SensitivityControl(QWidget):
    """Hamming distance threshold slider with spinbox."""

    value_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel("Sync Sensitivity:")
        label.setFixedWidth(130)
        label.setStyleSheet("font-weight: bold; color: #e0e0e0;")
        layout.addWidget(label)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(2, 40)
        self._slider.setValue(12)
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setTickInterval(2)
        self._slider.setStyleSheet(
            "QSlider::groove:horizontal { background: #2b2b2b; height: 8px; "
            "border-radius: 4px; border: 1px solid #555; }"
            "QSlider::handle:horizontal { background: #3a7bd5; width: 18px; "
            "margin: -5px 0; border-radius: 9px; }"
            "QSlider::sub-page:horizontal { background: #3a7bd5; border-radius: 4px; }"
        )
        layout.addWidget(self._slider, stretch=1)

        self._spinbox = QSpinBox()
        self._spinbox.setRange(2, 40)
        self._spinbox.setValue(12)
        self._spinbox.setFixedWidth(60)
        self._spinbox.setStyleSheet(
            "QSpinBox { background: #2b2b2b; color: #e0e0e0; border: 1px solid #555; "
            "border-radius: 4px; padding: 4px; }"
        )
        layout.addWidget(self._spinbox)

        hint = QLabel("(lower = stricter)")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(hint)

        self._slider.valueChanged.connect(self._spinbox.setValue)
        self._slider.valueChanged.connect(self.value_changed.emit)
        self._spinbox.valueChanged.connect(self._slider.setValue)

    def value(self) -> int:
        return self._slider.value()

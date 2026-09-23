"""Shared appearance colors and branding for Qt and rendered figures."""
from string import Template
from pathlib import Path
from PySide6 import QtCore as C, QtGui as G, QtWidgets as W


ASSET_DIR = Path(__file__).parent / "assets"

THEMES = {
    "Light": dict(bg="#f3f6fa", panel="#ffffff", fg="#18283d", muted="#506176",
                  accent="#1d5fa7", secondary="#6846b7", third="#a54479", cursor="#a95300",
                  border="#c7d2df", field="#ffffff", button="#e6edf5", hover="#d8e5ef",
                  disabled="#64748b", inactive="#e8edf3", selection="#dbeafe", on_accent="#ffffff"),
    "Dark": dict(bg="#0c1422", panel="#111e30", fg="#e7eef8", muted="#9aadc4",
                 accent="#66b3ff", secondary="#b69cff", third="#f2a0cf", cursor="#ffc577",
                 border="#30465f", field="#152438", button="#1b2b40", hover="#2b425c",
                 disabled="#8797ab", inactive="#142033", selection="#193b63", on_accent="#071a33"),
}


def colors(appearance="Light"):
    return THEMES[appearance]


def stylesheet(appearance="Light"):
    c = dict(colors(appearance), check=(ASSET_DIR / f"check-{appearance.lower()}.svg").as_posix())
    return Template("""QWidget { background: $bg; color: $fg; font-family: 'Helvetica Neue', 'Segoe UI'; font-size: 12px; }
QLabel#brand { font-size: 24px; font-weight: 700; letter-spacing: 1px; }
QLabel#subtitle { color: $muted; font-size: 12px; }
QLabel#sectionTitle { font-size: 19px; font-weight: 600; }
QGroupBox { border: 1px solid $border; border-radius: 10px; margin-top: 18px; padding: 15px 10px 8px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; top: 2px; color: $muted; }
QPushButton { background: $button; border: 1px solid $border; border-radius: 6px; padding: 8px 12px; }
QPushButton:hover, QToolButton:hover { background: $hover; border-color: $accent; }
QPushButton:disabled { color: $disabled; background: $inactive; }
QPushButton#primary { background: $accent; color: $on_accent; font-weight: 700; border: none; }
QPushButton#primary:disabled { background: $inactive; color: $disabled; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { background: $field; border: 1px solid $border; border-radius: 5px; padding: 5px; min-height: 19px; }
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: $accent; }
QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { color: $disabled; background: $inactive; }
QAbstractItemView, QPlainTextEdit, QTextEdit { background: $panel; color: $fg; border: 1px solid $border; border-radius: 5px; selection-background-color: $selection; selection-color: $fg; }
QListWidget::item { padding: 7px; }
QAbstractItemView::item:selected { background: $selection; color: $fg; }
QHeaderView::section { background: $button; color: $fg; padding: 6px; border: none; }
QTabBar::tab { padding: 9px 18px; background: $inactive; }
QTabBar::tab:selected { color: $accent; border-bottom: 2px solid $accent; }
QSlider::groove:horizontal { height: 5px; background: $border; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; margin: -5px 0; background: $accent; border-radius: 7px; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 12px; height: 12px; border: 1px solid $disabled; border-radius: 3px; background: $field; }
QCheckBox::indicator:checked { background: $accent; border-color: $accent; image: url("$check"); }
QStatusBar { color: $muted; border-top: 1px solid $border; }
QScrollArea, QToolBar { border: none; }
QToolButton:checked { background: $selection; }
QToolTip { background: $panel; color: $fg; border: 1px solid $border; padding: 5px; }
QProgressBar { border: 1px solid $border; border-radius: 4px; text-align: center; }
QProgressBar::chunk { background: $accent; }
""").substitute(c)


class FacilityLogo(W.QLabel):
    """Compact, theme-aware BaPSF wordmark rendered from the source PNG."""

    def __init__(self, appearance="Light", height=48, colorful=False, parent=None):
        super().__init__(parent)
        self.display_height = height
        self.colorful = colorful
        self.variant = ""
        self.setAlignment(C.Qt.AlignCenter)
        self.setSizePolicy(W.QSizePolicy.Fixed, W.QSizePolicy.Fixed)
        self.setAccessibleName("Basic Plasma Science Facility")
        self.setToolTip("Basic Plasma Science Facility")
        self.set_appearance(appearance)

    def set_appearance(self, appearance):
        variant = "White" if appearance == "Dark" else "Color" if self.colorful else "Black"
        self.variant = variant
        pixmap = G.QPixmap(str(ASSET_DIR / f"BaPSF_Logo+Name_{variant}_RGB.png"))
        if pixmap.isNull():
            self.setText("BaPSF")
            self.setFixedSize(90, self.display_height)
            return
        pixmap = pixmap.scaledToHeight(self.display_height, C.Qt.SmoothTransformation)
        self.setText("")
        self.setPixmap(pixmap)
        self.setFixedSize(pixmap.size())


def apply_appearance(app, appearance="Light"):
    """Set native-control colors too, including icons, checkboxes and dialogs."""
    c = colors(appearance)
    palette = G.QPalette()
    for role, key in {"Window": "bg", "WindowText": "fg", "Base": "panel", "AlternateBase": "inactive",
                      "Text": "fg", "Button": "button", "ButtonText": "fg", "ToolTipBase": "panel",
                      "ToolTipText": "fg", "Highlight": "selection", "HighlightedText": "fg",
                      "Link": "accent", "Light": "panel", "Midlight": "button", "Mid": "border",
                      "Dark": "border", "Shadow": "muted", "PlaceholderText": "muted"}.items():
        palette.setColor(getattr(G.QPalette.ColorRole, role), G.QColor(c[key]))
    for role in (G.QPalette.WindowText, G.QPalette.Text, G.QPalette.ButtonText):
        palette.setColor(G.QPalette.Disabled, role, G.QColor(c["disabled"]))
    app.setPalette(palette)
    app.setStyleSheet(stylesheet(appearance))

"""Shared desktop palette."""

STYLESHEET = """
QWidget { background: #101720; color: #dce6ef; font-family: 'Inter', 'Helvetica Neue', 'Segoe UI'; font-size: 13px; }
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget { background: #101720; }
QLabel#brand { font-size: 25px; font-weight: 700; color: #f2f7fc; }
QLabel#eyebrow { color: #53d6c7; font-size: 11px; font-weight: 700; }
QLabel#muted { color: #8c9daf; }
QLabel#metric { background: #192430; border: 1px solid #2a3848; border-radius: 9px; padding: 13px; font-size: 15px; }
QGroupBox { border: 1px solid #2a3848; border-radius: 8px; margin-top: 17px; padding: 13px 10px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #a7b8ca; }
QPushButton { background: #243342; border: 1px solid #364a5e; padding: 8px 12px; border-radius: 6px; font-weight: 600; }
QPushButton:hover { background: #30465a; border-color: #64839c; }
QPushButton:pressed { background: #38566b; }
QPushButton#plotTool { padding: 5px 9px; font-size: 12px; }
QPushButton#plotTool:checked { background: #28575d; color: #6fe4d4; border-color: #41c6b6; }
QPushButton#primary { background: #41c6b6; color: #0c2425; border: none; }
QPushButton#primary:hover { background: #65decf; }
QPushButton:disabled { color: #5f7182; background: #1c2732; border-color: #293745; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit { background: #1a2531; border: 1px solid #334456; padding: 6px; border-radius: 5px; selection-background-color: #267e7b; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border-color: #41c6b6; }
QComboBox QAbstractItemView { background: #1a2531; selection-background-color: #28575d; }
QTabWidget::pane { border: 1px solid #2a3848; border-radius: 6px; }
QTabBar::tab { color: #92a4b8; padding: 11px 14px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #61dfce; border-bottom: 2px solid #41c6b6; }
QSplitter::handle { background: #253140; width: 2px; }
QSlider::groove:horizontal { height: 5px; background: #33465a; border-radius: 2px; }
QSlider::handle:horizontal { width: 14px; margin: -5px 0; border-radius: 7px; background: #58d4c5; }
QCheckBox { spacing: 8px; }
QStatusBar { color: #91a5b8; border-top: 1px solid #293745; }
QToolTip { background: #243342; color: #f2f7fc; border: 1px solid #53697b; padding: 5px; }
"""

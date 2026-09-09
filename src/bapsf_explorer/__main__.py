import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="BaPSF HDF5 desktop explorer")
    parser.add_argument("file", nargs="?", help="BaPSF HDF5 file to inspect")
    parser.add_argument("--demo", action="store_true", help="Open a reproducible synthetic scan")
    args = parser.parse_args()
    from PyQt6.QtWidgets import QApplication
    from .window import ExplorerWindow
    app = QApplication(sys.argv[:1])
    app.setApplicationName("BaPSF Explorer")
    app.setOrganizationName("BaPSF Explorer")
    app.setStyle("Fusion")
    window = ExplorerWindow()
    window.show()
    if args.file:
        window.open_path(args.file)
    elif args.demo:
        window.load_demo()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

"""Cooperative cancellation at safe boundaries of read-only import work."""


class ImportCanceled(RuntimeError):
    """The import was stopped; no partial dataset may be accepted."""


def check_canceled(canceled):
    if canceled is not None and canceled():
        raise ImportCanceled("Import canceled.")

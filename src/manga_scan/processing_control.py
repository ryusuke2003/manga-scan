from pathlib import Path


class ProcessingCancelled(Exception):
    """Raised at safe checkpoints after a user requests cancellation."""


_CANCEL_FILE = "debug/cancel.request"


def cancel_request_path(project):
    return Path(project) / _CANCEL_FILE


def request_cancel(project):
    path = cancel_request_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def clear_cancel_request(project):
    cancel_request_path(project).unlink(missing_ok=True)


def cancel_requested(project):
    return cancel_request_path(project).exists()


def raise_if_cancelled(project):
    if cancel_requested(project):
        raise ProcessingCancelled("processing cancelled")

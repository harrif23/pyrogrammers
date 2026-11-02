"""The one error hierarchy, mapped to exit codes in one place."""

import sys

EXIT_OK, EXIT_ERROR, EXIT_STOPPED, EXIT_TRUNCATED = 0, 1, 2, 3


class evalsError(Exception):
    """Base class for every expected failure."""


class ConfigError(evalsError):
    """Missing or invalid local configuration."""


class DatasetError(evalsError):
    """Benchmark data is absent, altered, or fails validation."""


class ImageInputError(evalsError):
    """An image cannot be sent unmodified to the model."""


class RunStateError(evalsError):
    """A run's files conflict with the requested action."""


class ApiFailure(evalsError):
    """A model call produced no usable response. Never counted as a model answer."""

    def __init__(self, kind, message, status=None, systematic=False, latency_seconds=None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status
        self.systematic = systematic
        self.latency_seconds = latency_seconds

    def to_dict(self):
        return {"kind": self.kind, "status": self.status, "message": self.message,
                "systematic": self.systematic}


def run_guarded(func, *args):
    """Run a command, turning expected failures into a message and an exit code."""
    try:
        return func(*args)
    except evalsError as error:
        print(f"Error: {error}", file=sys.stderr)
        return EXIT_ERROR

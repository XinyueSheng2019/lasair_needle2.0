"""
Shared log used across the project. Execution order is preserved by flushing
after each write.

Usage:
  - In your main script (e.g. generate_annotator.py):
      import logs
      logs.set_log(open('logs/log_...', 'w'))
      # ... use logs.log.write('message\\n') anywhere ...
      logs.close_log()

  - In any other module:
      import logs
      logs.log.write('message\\n')   # same log, execution order preserved
"""

from datetime import datetime


class _LogWriter:
    """Wraps a file so every write() is flushed, preserving execution order."""

    def __init__(self, file_handle, add_timestamp=False):
        self._file = file_handle
        self._add_timestamp = add_timestamp

    def write(self, msg):
        if self._file is None:
            return
        if self._add_timestamp and msg.strip():
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            msg = f"[{ts}] {msg}"
        self._file.write(msg)
        self._file.flush()

    def close(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None


# Public log: use log.write('message\n') from any module. Set via set_log().
log = None


def set_log(file_handle, add_timestamp=True):
    """Set the global log. Call once at startup from the main script."""
    global log
    log = _LogWriter(file_handle, add_timestamp=add_timestamp)


def close_log():
    """Close the log file. Call from main script at exit."""
    global log
    if log is not None:
        log.close()
        log = None

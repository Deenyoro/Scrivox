"""Thread-safe stdout redirect into the GUI log, line-buffered."""

import io
import threading


class LogRedirect(io.TextIOBase):
    """Routes stdout writes into LogFrame's batched append, one line at a time.

    Buffers partial writes until a newline arrives and collapses \r progress
    updates (tqdm-style) to their final state, so chatty libraries can't
    flood the Tk event queue with per-write callbacks, bypass LogFrame's
    line cap, or yank the user's scrollback with forced scrolls.
    """

    def __init__(self, log_frame, root, original_stdout=None):
        super().__init__()
        self.log_frame = log_frame
        self.root = root
        self.original_stdout = original_stdout
        self._lock = threading.Lock()
        self._partial = ""

    def write(self, text):
        if not text:
            return 0
        # Also write to original stdout (visible in console)
        if self.original_stdout:
            try:
                self.original_stdout.write(text)
                self.original_stdout.flush()
            except Exception:
                pass

        lines_out = []
        with self._lock:
            self._partial += text
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                # A line overwritten by \r progress updates only matters in
                # its final state
                line = line.split("\r")[-1]
                if line:
                    lines_out.append(line + "\n")
            # Don't let a never-newline-terminated stream grow unbounded
            if len(self._partial) > 8192:
                lines_out.append(self._partial.split("\r")[-1] + "\n")
                self._partial = ""

        if lines_out:
            try:
                # LogFrame.append batches inserts, enforces the line cap, and
                # preserves scroll position — one after() per write, not per
                # widget operation
                self.root.after(0, self.log_frame.append, "".join(lines_out))
            except Exception:
                pass  # widget may be destroyed

        return len(text)

    def flush(self):
        if self.original_stdout:
            try:
                self.original_stdout.flush()
            except Exception:
                pass

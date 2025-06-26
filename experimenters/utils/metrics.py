# experimenters/utils/metrics.py
import time


class ThroughputMeter:
    """
    Rolling tokens/sec counter.
    Keeps the last `window` measurements and exposes `rate` property.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self.counter = 0
        self.start = time.perf_counter()
        self.buffer = []  # list of (tokens, elapsed)

    def update(self, n_tokens: int):
        now = time.perf_counter()
        elapsed = now - self.start
        self.buffer.append((n_tokens, elapsed))
        self.counter += n_tokens
        self.start = now

        # clip buffer
        if len(self.buffer) > self.window:
            self.buffer.pop(0)

    @property
    def rate(self) -> float:
        if not self.buffer:
            return 0.0
        toks, secs = zip(*self.buffer)
        return sum(toks) / sum(secs)

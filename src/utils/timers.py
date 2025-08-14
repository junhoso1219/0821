from __future__ import annotations
import time
class Timer:
    def __enter__(self): self.t0 = time.time(); return self
    def __exit__(self, *args): self.t1 = time.time()
    @property
    def elapsed(self): return self.t1 - self.t0 if hasattr(self,'t1') else time.time() - self.t0

from __future__ import annotations
class EMA:
    def __init__(self, beta: float, init: float | None = None):
        assert 0.0 < beta < 1.0
        self.beta = beta
        self.value = init
        self.ready = init is not None
    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x; self.ready = True
        else:
            self.value = self.beta * self.value + (1 - self.beta) * x
        return self.value
    def get(self) -> float | None:
        return self.value

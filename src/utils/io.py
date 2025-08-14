from __future__ import annotations
import csv, os
from typing import Dict, Any

class CSVLogger:
    def __init__(self, path: str, fieldnames: list[str]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.writer.writeheader(); self.f.flush()
    def log(self, row: Dict[str, Any]):
        self.writer.writerow(row); self.f.flush()
    def close(self):
        try: self.f.close()
        except: pass

"""
CSV logger for Decision's per-frame output, mirroring Hafsa's
PerceptionLogger so Raks has the same kind of file from both
modules to align by frame_id for her ablation study.
"""

import csv
import os


class DecisionLogger:
    def __init__(self, filepath="decision/output_log.csv"):
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.filepath = filepath
        self.file = open(filepath, "w", newline="")
        self.writer = None

    def log(self, frame_dict):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=frame_dict.keys())
            self.writer.writeheader()
        self.writer.writerow(frame_dict)

    def close(self):
        self.file.close()
        print(f"Log saved: {self.filepath}")

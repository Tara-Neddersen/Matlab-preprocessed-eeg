#!/usr/bin/env python3

import os

# Force Matplotlib WebAgg backend with a fixed port and open to all interfaces
os.environ.setdefault("MPLBACKEND", "WebAgg")

import matplotlib
matplotlib.use("WebAgg")
matplotlib.rcParams["webagg.address"] = "0.0.0.0"
matplotlib.rcParams["webagg.port"] = 8988

import matplotlib.pyplot as plt

from eeg_viewer import EEGViewer

if __name__ == "__main__":
    print("=== EEG Viewer Web Launcher (WebAgg) ===")
    print("- Backend: WebAgg")
    print("- Address: http://0.0.0.0:8988/")
    print("  (Your environment may show a forwarded/preview link for this port)")

    viewer = EEGViewer()
    # Block here to keep the Tornado server alive and serve the UI
    viewer.run()
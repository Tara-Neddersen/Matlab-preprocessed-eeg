#!/usr/bin/env python3
"""
Simple launcher for the EEG viewer
Checks if data exists and runs the viewer
"""

import os
import sys

def check_data():
    """Check if processed data exists"""
    data_file = 'processed_data/eeg_data.csv'
    
    if not os.path.exists(data_file):
        print("❌ Processed data not found!")
        print(f"   Looking for: {data_file}")
        print("\n📋 Please run the MATLAB script first:")
        print("   1. Open MATLAB")
        print("   2. Navigate to this folder")
        print("   3. Run: matlab_data_processor()")
        print("\n   This will create the processed data files.")
        return False
    
    print("✅ Processed data found!")
    return True

def main():
    """Main launcher function"""
    print("=== EEG Viewer Launcher ===")
    
    # Check if data exists
    if not check_data():
        return
    
    # Check if required packages are available
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy import signal
        print("✅ All required packages available")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("   Please install: pip install numpy pandas matplotlib scipy")
        return
    
    # Launch the viewer
    print("\n🚀 Starting EEG viewer...")
    print("   Close the viewer window to exit")
    
    try:
        from eeg_viewer import EEGViewer
        viewer = EEGViewer()
        viewer.run()
    except Exception as e:
        print(f"❌ Error starting viewer: {e}")
        print("   Check the console for more details")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Interactive EEG Viewer with Seizure Detection
Loads MATLAB-processed EEG data and provides interactive visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, TextBox
import matplotlib.patches as patches
from scipy.signal import find_peaks, butter, lfilter, filtfilt, iirnotch
import scipy.signal as signal
import os
import json
from datetime import datetime

class EEGViewer:
    def __init__(self, data_file='processed_data/eeg_data.csv'):
        """Initialize the EEG viewer"""
        self.data_file = data_file
        self.seizures = []
        self.current_seizure = None
        self.detection_mode = False
        
        # Signal processing state variables
        self.filter_enabled = False
        self.scale_enabled = False
        self.baseline_enabled = False
        self.notch_enabled = False
        self.original_data = None
        self.processed_data = None
        
        # MATLAB-style display variables
        self.frequency_display = False
        self.signal_scaling = False
        self.baseline_display = False
        self.grid_enabled = True
        self.line_thickness = 0.5
        self.zoom_level = 1.0
        
        # Real MATLAB EEG interface variables
        self.amplitude_scale = 1.0  # EEGmax equivalent
        self.marking_mode = None  # 'seizure', 'artifact', or None
        self.current_marking = None  # for marking start/end
        self.marking_type = 'seizure'  # 'seizure' or 'artifact'
        
        self.load_data()
        self.setup_plot()
        self.setup_controls()
        self.update_plot()  # Initial plot update
        
    def load_data(self):
        """Load the MATLAB-processed EEG data"""
        print("Loading EEG data...")
        
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
        
        # Load data
        df = pd.read_csv(self.data_file)
        self.times = df['Time_s'].values
        self.data = np.column_stack([df['Channel_57_mV'].values, df['Channel_61_mV'].values])
        self.fs = 1 / (self.times[1] - self.times[0])
        
        # Store original data for processing
        self.original_data = self.data.copy()
        self.processed_data = self.data.copy()
        
        print(f"Data loaded: {self.data.shape[0]} samples, {self.fs:.1f} Hz")
        print(f"Duration: {self.times[-1]/60:.1f} minutes")
        
        # Initialize viewing parameters
        self.window_size = 30  # seconds
        self.current_time = 0
        self.update_view_range()
        
    def setup_plot(self):
        """Setup the main plot with separate subplots for each channel"""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        self.fig.suptitle('Interactive EEG Viewer - MATLAB Processed Data', fontsize=16)
        
        # Plot channels in separate subplots
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=0.5, label='Channel 57')
        self.line2, = self.ax2.plot([], [], 'r-', linewidth=0.5, label='Channel 61')
        
        # Setup plot properties
        self.ax1.set_ylabel('Channel 57 (mV)')
        self.ax1.grid(True, alpha=0.3)
        self.ax1.legend()
        
        self.ax2.set_xlabel('Time (seconds)')
        self.ax2.set_ylabel('Channel 61 (mV)')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.legend()
        
        # Add seizure highlighting for both subplots
        self.seizure_patches1 = []
        self.seizure_patches2 = []
        
        # Setup keyboard shortcuts
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Update plot
        self.update_plot()
        
    def setup_controls(self):
        """Setup interactive controls"""
        # Create control panel
        control_height = 0.25  # Increased for two rows of buttons
        self.fig.subplots_adjust(bottom=control_height)
        
        # Navigation buttons (top row)
        btn_width = 0.08
        btn_height = 0.04
        btn_y = 0.15
        
        # Previous button
        ax_prev = plt.axes([0.1, btn_y, btn_width, btn_height])
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_prev.on_clicked(self.previous_window)
        
        # Next button
        ax_next = plt.axes([0.2, btn_y, btn_width, btn_height])
        self.btn_next = Button(ax_next, 'Next')
        self.btn_next.on_clicked(self.next_window)
        
        # Jump to time button
        ax_jump = plt.axes([0.3, btn_y, btn_width, btn_height])
        self.btn_jump = Button(ax_jump, 'Jump to Time')
        self.btn_jump.on_clicked(self.jump_to_time)
        
        # Time input box
        ax_time = plt.axes([0.4, btn_y, 0.1, btn_height])
        self.time_input = TextBox(ax_time, 'Time (s):', initial='0')
        
        # Seizure detection buttons
        ax_detect = plt.axes([0.55, btn_y, btn_width, btn_height])
        self.btn_detect = Button(ax_detect, 'Auto Detect')
        self.btn_detect.on_clicked(self.auto_detect_seizures)
        
        ax_manual = plt.axes([0.65, btn_y, btn_width, btn_height])
        self.btn_manual = Button(ax_manual, 'Manual Mark')
        self.btn_manual.on_clicked(self.manual_mark_seizure)
        
        # Clear seizures button
        ax_clear = plt.axes([0.75, btn_y, btn_width, btn_height])
        self.btn_clear = Button(ax_clear, 'Clear All')
        self.btn_clear.on_clicked(self.clear_seizures)
        
        # Save button
        ax_save = plt.axes([0.85, btn_y, btn_width, btn_height])
        self.btn_save = Button(ax_save, 'Save Results')
        self.btn_save.on_clicked(self.save_results)
        
        # Signal processing buttons (bottom row)
        btn_y2 = 0.08
        
        # Filter buttons
        ax_filter = plt.axes([0.1, btn_y2, btn_width, btn_height])
        self.btn_filter = Button(ax_filter, 'Filter (F)')
        self.btn_filter.on_clicked(self.toggle_filter)
        
        # Scale buttons
        ax_scale = plt.axes([0.2, btn_y2, btn_width, btn_height])
        self.btn_scale = Button(ax_scale, 'Scale (S)')
        self.btn_scale.on_clicked(self.toggle_scale)
        
        # Baseline correction
        ax_baseline = plt.axes([0.3, btn_y2, btn_width, btn_height])
        self.btn_baseline = Button(ax_baseline, 'Baseline (B)')
        self.btn_baseline.on_clicked(self.toggle_baseline)
        
        # Notch filter (50/60 Hz)
        ax_notch = plt.axes([0.4, btn_y2, btn_width, btn_height])
        self.btn_notch = Button(ax_notch, 'Notch (N)')
        self.btn_notch.on_clicked(self.toggle_notch)
        
        # Reset all processing
        ax_reset = plt.axes([0.5, btn_y2, btn_width, btn_height])
        self.btn_reset = Button(ax_reset, 'Reset (R)')
        self.btn_reset.on_clicked(self.reset_processing)
        
        # Window size slider
        ax_slider = plt.axes([0.1, 0.02, 0.3, 0.02])
        self.slider_window = Slider(ax_slider, 'Window (s)', 5, 120, valinit=self.window_size)
        self.slider_window.on_changed(self.update_window_size)
        
        # Status text
        self.status_text = self.fig.text(0.5, 0.06, 'Ready', ha='center', fontsize=10)
        
    def update_view_range(self):
        """Update the current viewing range"""
        start_idx = int(self.current_time * self.fs)
        end_idx = int((self.current_time + self.window_size) * self.fs)
        
        # Ensure indices are within bounds
        start_idx = max(0, min(start_idx, len(self.times) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(self.times)))
        
        self.view_start = start_idx
        self.view_end = end_idx
        
    def update_plot(self):
        """Update the plot with current data"""
        # Get current view data
        view_times = self.times[self.view_start:self.view_end]
        view_data1 = self.data[self.view_start:self.view_end, 0]
        view_data2 = self.data[self.view_start:self.view_end, 1]
        
        # Update lines
        self.line1.set_data(view_times, view_data1)
        self.line2.set_data(view_times, view_data2)
        
        # Update axis limits
        self.ax1.set_xlim(view_times[0], view_times[-1])
        self.ax2.set_xlim(view_times[0], view_times[-1])
        
        # Apply amplitude scaling (like MATLAB's EEGmax)
        max_amplitude = self.amplitude_scale * 0.4  # Base amplitude like MATLAB
        self.ax1.set_ylim(-max_amplitude * 3, max_amplitude)
        self.ax2.set_ylim(-max_amplitude * 3, max_amplitude)
        
        # Update seizure patches
        self.update_seizure_patches()
        
        # Update status
        self.update_status()
        
        self.fig.canvas.draw_idle()
        
    def update_seizure_patches(self):
        """Update seizure highlighting patches for both subplots"""
        # Clear existing patches
        for patch in self.seizure_patches1:
            patch.remove()
        self.seizure_patches1.clear()
        
        for patch in self.seizure_patches2:
            patch.remove()
        self.seizure_patches2.clear()
        
        # Add patches for seizures in current view
        for seizure in self.seizures:
            start_time, end_time = seizure
            if (start_time <= self.times[self.view_end-1] and 
                end_time >= self.times[self.view_start]):
                
                # Clip to current view
                patch_start = max(start_time, self.times[self.view_start])
                patch_end = min(end_time, self.times[self.view_end-1])
                
                # Create patches for both subplots
                ylim1 = self.ax1.get_ylim()
                rect1 = patches.Rectangle(
                    (patch_start, ylim1[0]), 
                    patch_end - patch_start, 
                    ylim1[1] - ylim1[0],
                    alpha=0.3, color='red', label='Seizure'
                )
                self.ax1.add_patch(rect1)
                self.seizure_patches1.append(rect1)
                
                ylim2 = self.ax2.get_ylim()
                rect2 = patches.Rectangle(
                    (patch_start, ylim2[0]), 
                    patch_end - patch_start, 
                    ylim2[1] - ylim2[0],
                    alpha=0.3, color='red', label='Seizure'
                )
                self.ax2.add_patch(rect2)
                self.seizure_patches2.append(rect2)
        
    def update_status(self):
        """Update status text"""
        if hasattr(self, 'status_text'):
            status = f"Time: {self.current_time:.1f}s | Window: {self.window_size}s | Seizures: {len(self.seizures)}"
            self.status_text.set_text(status)
        
    def previous_window(self, event):
        """Move to previous time window"""
        self.current_time = max(0, self.current_time - self.window_size)
        self.update_view_range()
        self.update_plot()
        
    def next_window(self, event):
        """Move to next time window"""
        max_time = self.times[-1] - self.window_size
        self.current_time = min(max_time, self.current_time + self.window_size)
        self.update_view_range()
        self.update_plot()
        
    def jump_to_time(self, event):
        """Jump to specific time"""
        try:
            time = float(self.time_input.text)
            if 0 <= time <= self.times[-1] - self.window_size:
                self.current_time = time
                self.update_view_range()
                self.update_plot()
            else:
                self.status_text.set_text(f"Invalid time: {time}")
        except ValueError:
            self.status_text.set_text("Invalid time format")
            
    def update_window_size(self, val):
        """Update window size from slider"""
        self.window_size = val
        self.update_view_range()
        self.update_plot()
        
    def auto_detect_seizures(self, event):
        """Automatically detect seizures"""
        self.status_text.set_text("Detecting seizures...")
        self.fig.canvas.draw_idle()
        
        # Simple seizure detection algorithm
        detected_seizures = self.detect_seizures()
        
        # Add new seizures (avoid duplicates)
        for seizure in detected_seizures:
            if not self.is_seizure_overlapping(seizure):
                self.seizures.append(seizure)
        
        self.seizures.sort()  # Sort by start time
        self.update_plot()
        self.status_text.set_text(f"Detected {len(detected_seizures)} seizures")
        
    def detect_seizures(self):
        """Simple seizure detection algorithm"""
        seizures = []
        
        # Parameters for detection
        threshold = 2.0  # mV
        min_duration = 2.0  # seconds
        min_peaks = 5
        
        for ch_idx in range(self.data.shape[1]):
            signal = self.data[:, ch_idx]
            
            # Find peaks above threshold
            peaks, properties = find_peaks(np.abs(signal), height=threshold, distance=int(self.fs * 0.5))
            
            if len(peaks) < min_peaks:
                continue
                
            # Group peaks into potential seizures
            peak_times = self.times[peaks]
            current_start = peak_times[0]
            current_end = peak_times[0]
            
            for i in range(1, len(peak_times)):
                if peak_times[i] - peak_times[i-1] < 5.0:  # 5 second gap threshold
                    current_end = peak_times[i]
                else:
                    # Check if current group is long enough
                    if current_end - current_start >= min_duration:
                        seizures.append((current_start, current_end))
                    current_start = peak_times[i]
                    current_end = peak_times[i]
            
            # Check last group
            if current_end - current_start >= min_duration:
                seizures.append((current_start, current_end))
        
        return seizures
        
    def manual_mark_seizure(self, event):
        """Enter manual seizure marking mode"""
        self.detection_mode = True
        self.status_text.set_text("Click to mark seizure start and end")
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def on_click(self, event):
        """Handle mouse clicks for manual seizure marking"""
        if not self.detection_mode or event.inaxes not in [self.ax1, self.ax2]:
            return
            
        click_time = event.xdata
        
        if self.current_seizure is None:
            # Start marking
            self.current_seizure = [click_time, None]
            self.status_text.set_text(f"Seizure start: {click_time:.1f}s. Click for end.")
        else:
            # End marking
            self.current_seizure[1] = click_time
            start_time, end_time = self.current_seizure
            
            if end_time > start_time:
                self.seizures.append((start_time, end_time))
                self.seizures.sort()
                self.status_text.set_text(f"Seizure marked: {start_time:.1f}s - {end_time:.1f}s")
            else:
                self.status_text.set_text("Invalid seizure: end before start")
            
            self.current_seizure = None
            self.detection_mode = False
            self.fig.canvas.mpl_disconnect('button_press_event')
            self.update_plot()
            
    def is_seizure_overlapping(self, new_seizure):
        """Check if a seizure overlaps with existing ones"""
        new_start, new_end = new_seizure
        
        for existing_seizure in self.seizures:
            existing_start, existing_end = existing_seizure
            
            # Check for overlap
            if (new_start <= existing_end and new_end >= existing_start):
                return True
                
        return False
        
    def clear_seizures(self, event):
        """Clear all marked seizures"""
        self.seizures.clear()
        self.update_plot()
        self.status_text.set_text("All seizures cleared")
        
    def save_results(self, event):
        """Save seizure detection results"""
        if not self.seizures:
            self.status_text.set_text("No seizures to save")
            return
            
        # Create results directory
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # Save seizures to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f'seizures_{timestamp}.json')
        
        results = {
            'timestamp': timestamp,
            'data_file': self.data_file,
            'total_seizures': len(self.seizures),
            'seizures': self.seizures,
            'detection_parameters': {
                'window_size': self.window_size,
                'sample_rate': self.fs
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Save summary
        summary_file = os.path.join(results_dir, f'summary_{timestamp}.txt')
        with open(summary_file, 'w') as f:
            f.write("EEG Seizure Detection Results\n")
            f.write("=============================\n\n")
            f.write(f"Data file: {self.data_file}\n")
            f.write(f"Analysis date: {timestamp}\n")
            f.write(f"Total seizures detected: {len(self.seizures)}\n")
            f.write(f"Recording duration: {self.times[-1]/60:.1f} minutes\n\n")
            f.write("Seizure Details:\n")
            f.write("Start (s)\tEnd (s)\tDuration (s)\n")
            f.write("-" * 40 + "\n")
            
            for i, (start, end) in enumerate(self.seizures, 1):
                duration = end - start
                f.write(f"{start:.1f}\t\t{end:.1f}\t\t{duration:.1f}\n")
                
        self.status_text.set_text(f"Results saved: {results_file}")
        print(f"Results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")
        
    # Signal processing functions
    def toggle_filter(self, event):
        """Toggle additional filtering (high-pass 2Hz, low-pass 20Hz)"""
        self.filter_enabled = not self.filter_enabled
        self.apply_processing()
        status = "ON" if self.filter_enabled else "OFF"
        self.status_text.set_text(f"Filter {status}")
        
    def toggle_scale(self, event):
        """Toggle signal scaling (normalize to ±1)"""
        self.scale_enabled = not self.scale_enabled
        self.apply_processing()
        status = "ON" if self.scale_enabled else "OFF"
        self.status_text.set_text(f"Scale {status}")
        
    def toggle_baseline(self, event):
        """Toggle baseline correction (remove DC offset)"""
        self.baseline_enabled = not self.baseline_enabled
        self.apply_processing()
        status = "ON" if self.baseline_enabled else "OFF"
        self.status_text.set_text(f"Baseline {status}")
        
    def toggle_notch(self, event):
        """Toggle notch filter (50/60 Hz)"""
        self.notch_enabled = not self.notch_enabled
        self.apply_processing()
        status = "ON" if self.notch_enabled else "OFF"
        self.status_text.set_text(f"Notch {status}")
        
    def reset_processing(self, event):
        """Reset all signal processing"""
        self.filter_enabled = False
        self.scale_enabled = False
        self.baseline_enabled = False
        self.notch_enabled = False
        self.apply_processing()
        self.status_text.set_text("All processing reset")
        
    def apply_processing(self):
        """Apply all enabled signal processing"""
        # Start with original data
        self.processed_data = self.original_data.copy()
        
        # Apply baseline correction first
        if self.baseline_enabled:
            for ch in range(self.processed_data.shape[1]):
                self.processed_data[:, ch] = self.processed_data[:, ch] - np.mean(self.processed_data[:, ch])
        
        # Apply notch filter (50/60 Hz)
        if self.notch_enabled:
            for ch in range(self.processed_data.shape[1]):
                # Design notch filter at 50 Hz (adjust for 60 Hz if needed)
                notch_freq = 50.0  # Hz
                Q = 30.0  # Quality factor
                w0 = notch_freq / (self.fs / 2)
                b, a = signal.iirnotch(w0, Q)
                self.processed_data[:, ch] = signal.filtfilt(b, a, self.processed_data[:, ch])
        
        # Apply additional filtering
        if self.filter_enabled:
            for ch in range(self.processed_data.shape[1]):
                # High-pass filter at 2 Hz
                b_hp, a_hp = signal.butter(2, 2.0 / (self.fs / 2), btype='high')
                self.processed_data[:, ch] = signal.filtfilt(b_hp, a_hp, self.processed_data[:, ch])
                
                # Low-pass filter at 20 Hz
                b_lp, a_lp = signal.butter(2, 20.0 / (self.fs / 2), btype='low')
                self.processed_data[:, ch] = signal.filtfilt(b_lp, a_lp, self.processed_data[:, ch])
        
        # Apply scaling
        if self.scale_enabled:
            for ch in range(self.processed_data.shape[1]):
                max_val = np.max(np.abs(self.processed_data[:, ch]))
                if max_val > 0:
                    self.processed_data[:, ch] = self.processed_data[:, ch] / max_val
        
        # Update the data used for plotting
        self.data = self.processed_data.copy()
        self.update_plot()
        
    def on_key_press(self, event):
        """Handle keyboard shortcuts (Real MATLAB EEG interface)"""
        if event.key.lower() == 'n':
            # N - Next window
            self.next_window(None)
        elif event.key.lower() == 'p':
            # P - Previous window
            self.previous_window(None)
        elif event.key.lower() == 'w':
            # W - Widen window (increase time window)
            self.widen_window()
        elif event.key.lower() == 't':
            # T - Tighten window (decrease time window)
            self.tighten_window()
        elif event.key.lower() == 'u':
            # U - Up (decrease amplitude scale)
            self.decrease_amplitude()
        elif event.key.lower() == 'd':
            # D - Down (increase amplitude scale)
            self.increase_amplitude()
        elif event.key.lower() == 'j':
            # J - Jump to time
            self.time_input.set_focus()
        elif event.key.lower() == 's':
            # S - Seizure start marking
            self.manual_mark_seizure(None)
        elif event.key.lower() == 'a':
            # A - Artifact start marking
            self.manual_mark_artifact(None)
        elif event.key.lower() == 'e':
            # E - End marking (for seizure/artifact)
            self.end_marking()
        elif event.key.lower() == 'x':
            # X - Exchange type (switch between seizure/artifact)
            self.exchange_marking_type()
        elif event.key.lower() == 'q':
            # Q - Quit
            plt.close(self.fig)
        elif event.key.lower() == 'h':
            # H - Show help
            self.show_help()
    
    # Real MATLAB EEG interface functions
    def widen_window(self):
        """W - Widen window (increase time window)"""
        self.window_size = min(120, self.window_size * 1.5)
        self.update_view_range()
        self.update_plot()
        self.status_text.set_text(f"Window widened: {self.window_size:.1f}s")
        
    def tighten_window(self):
        """T - Tighten window (decrease time window)"""
        self.window_size = max(5, self.window_size / 1.5)
        self.update_view_range()
        self.update_plot()
        self.status_text.set_text(f"Window tightened: {self.window_size:.1f}s")
        
    def decrease_amplitude(self):
        """U - Up (decrease amplitude scale - zoom in)"""
        self.amplitude_scale *= 0.5
        self.update_plot()
        self.status_text.set_text(f"Amplitude scale decreased: {self.amplitude_scale:.2f}")
        
    def increase_amplitude(self):
        """D - Down (increase amplitude scale - zoom out)"""
        self.amplitude_scale *= 2.0
        self.update_plot()
        self.status_text.set_text(f"Amplitude scale increased: {self.amplitude_scale:.2f}")
        
    def manual_mark_artifact(self, event):
        """A - Artifact start marking"""
        self.marking_mode = 'artifact'
        self.marking_type = 'artifact'
        self.status_text.set_text("Click to mark artifact start")
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
    def end_marking(self):
        """E - End marking (for seizure/artifact)"""
        if self.current_marking is not None:
            self.marking_mode = 'end'
            self.status_text.set_text("Click to mark end")
            self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        else:
            self.status_text.set_text("No start marking to end")
            
    def exchange_marking_type(self):
        """X - Exchange type (switch between seizure/artifact)"""
        if self.marking_type == 'seizure':
            self.marking_type = 'artifact'
            self.status_text.set_text("Marking type: ARTIFACT")
        else:
            self.marking_type = 'seizure'
            self.status_text.set_text("Marking type: SEIZURE")
        
    def show_help(self):
        """Show keyboard shortcuts help"""
        help_text = """
Real MATLAB EEG Interface Shortcuts:
====================================
Navigation:
  N - Next window
  P - Previous window
  J - Jump to time
  W - Widen window (increase time window)
  T - Tighten window (decrease time window)

Amplitude Control:
  U - Up (decrease amplitude scale - zoom in)
  D - Down (increase amplitude scale - zoom out)

Marking:
  S - Seizure start marking
  A - Artifact start marking
  E - End marking (for seizure/artifact)
  X - Exchange type (switch between seizure/artifact)

Other:
  Q - Quit
  H - Show this help
        """
        print(help_text)
        self.status_text.set_text("Help shown in console")
        
    def run(self):
        """Run the EEG viewer"""
        plt.show()

def main():
    """Main function to run the EEG viewer"""
    print("=== Interactive EEG Viewer ===")
    print("Loading MATLAB-processed EEG data...")
    
    try:
        viewer = EEGViewer()
        print("EEG viewer started successfully!")
        print("\nControls:")
        print("- Previous/Next: Navigate through time")
        print("- Jump to Time: Enter specific time")
        print("- Auto Detect: Automatically detect seizures")
        print("- Manual Mark: Click to mark seizures manually")
        print("- Clear All: Remove all marked seizures")
        print("- Save Results: Save detection results")
        print("\nStarting viewer...")
        
        viewer.run()
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to run the MATLAB script 'matlab_data_processor.m' first")
        print("to generate the processed data files.")

if __name__ == "__main__":
    main() 
import tkinter as tk
from tkinter import Canvas
import numpy as np
import math
import time

class SignalLocalizationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Localization UI")
        
        # Canvas for compass display
        self.canvas = Canvas(root, width=400, height=400, bg="white")
        self.canvas.pack()
        
        # Draw grid background
        self.draw_grid()
        
        # Create compass markings
        self.center_x, self.center_y = 200, 200
        self.radius = 120
        self.draw_compass()
        
        # Create directional arrows
        self.arrows = []  # Store arrows for measurements
        self.main_arrow = self.canvas.create_line(self.center_x, self.center_y, 
                                                  self.center_x, self.center_y - 100, 
                                                  arrow=tk.LAST, width=5, fill="blue")
        self.current_angle = 0  # Track arrow angle
        
        # Signal strength display
        self.signal_label = tk.Label(root, text="Signal Strength: - dB", font=("Arial", 14))
        self.signal_label.pack(pady=5)
        
        # Distance estimation display
        self.distance_label = tk.Label(root, text="Estimated Distance: - m", font=("Arial", 14))
        self.distance_label.pack(pady=5)
        
        # Live data feed display
        self.data_feed = tk.Text(root, height=6, width=50, state=tk.DISABLED, font=("Arial", 10))
        self.data_feed.pack(pady=5)
        
        # Buttons
        self.measure_button = tk.Button(root, text="Measure", command=self.measure_direction)
        self.measure_button.pack(pady=5)
        
        self.perform_button = tk.Button(root, text="Perform Algorithm", command=self.perform_algorithm)
        self.perform_button.pack(pady=5)
        
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_measurements)
        self.clear_button.pack(pady=5)
        
        # Store measurements
        self.measurements = []
        self.ml_prediction = (0, 0)
        self.longitude = 0
        self.latitude = 0
        self.signal_strength = 0
        
        # Update UI every second
        self.update_ui()
    
    def draw_grid(self):
        """Draws a grid in the background"""
        grid_spacing = 40  # Space between grid lines
        for x in range(0, 400, grid_spacing):
            self.canvas.create_line(x, 0, x, 400, fill="lightgray")
        for y in range(0, 400, grid_spacing):
            self.canvas.create_line(0, y, 400, y, fill="lightgray")
    
    def draw_compass(self):
        """Draws compass markings and labels"""
        directions = {0: "N", 90: "E", 180: "S", 270: "W"}  # Correct placement of cardinal directions
        for angle in range(0, 360, 30):  # Markings every 30 degrees
            radian = np.radians(-angle + 90)  # Adjust angle so North is at the top
            x1 = self.center_x + (self.radius - 10) * np.cos(radian)
            y1 = self.center_y - (self.radius - 10) * np.sin(radian)
            x2 = self.center_x + self.radius * np.cos(radian)
            y2 = self.center_y - self.radius * np.sin(radian)
            self.canvas.create_line(x1, y1, x2, y2, fill="black")
            
            text_x = self.center_x + (self.radius + 20) * np.cos(radian)
            text_y = self.center_y - (self.radius + 20) * np.sin(radian)
            label_text = directions.get(angle, str(angle))
            self.canvas.create_text(text_x, text_y, text=label_text, font=("Arial", 12, "bold"), fill="black")
    
    def measure_direction(self):
        """Capture direction and signal strength when Measure button is clicked."""
        angle = self.current_angle  # Use last known angle instead of randomization
        signal_strength = self.signal_strength  # Use real input data
        estimated_distance = self.estimate_distance(signal_strength)
        self.measurements.append((angle, signal_strength))
        
        # Draw arrow for this measurement
        length = 100
        end_x = self.center_x + length * np.cos(np.radians(-angle + 90))
        end_y = self.center_y - length * np.sin(np.radians(-angle + 90))
        arrow = self.canvas.create_line(self.center_x, self.center_y, end_x, end_y, arrow=tk.LAST, width=3, fill="red")
        self.arrows.append(arrow)
        
        # Update labels
        self.signal_label.config(text=f"Signal Strength: {signal_strength} dB")
        self.distance_label.config(text=f"Estimated Distance: {estimated_distance} m")
        
        # Log measurement to data feed
        self.data_feed.config(state=tk.NORMAL)
        self.data_feed.insert(tk.END, f"Measured at {angle}°: {signal_strength} dB, Distance: {estimated_distance} m\n")
        self.data_feed.see(tk.END)
        self.data_feed.config(state=tk.DISABLED)
    
    def perform_algorithm(self):
        """Processes measurements and provides direction to step in."""
        if len(self.measurements) >= 3:
            best_direction = sum(angle for angle, _ in self.measurements) // len(self.measurements)
            self.animate_arrow(best_direction)
            
            # Log algorithm result to data feed
            self.data_feed.config(state=tk.NORMAL)
            self.data_feed.insert(tk.END, f"Recommended step direction: {best_direction}°\n")
            self.data_feed.see(tk.END)
            self.data_feed.config(state=tk.DISABLED)
    
    def animate_arrow(self, target_angle):
        """Smoothly moves the arrow towards the target angle."""
        length = 100
        end_x = self.center_x + length * np.cos(np.radians(-target_angle + 90))
        end_y = self.center_y - length * np.sin(np.radians(-target_angle + 90))
        self.canvas.coords(self.main_arrow, self.center_x, self.center_y, end_x, end_y)
        self.current_angle = target_angle
    
    def clear_measurements(self):
        """Clears all arrows and resets measurement list."""
        for arrow in self.arrows:
            self.canvas.delete(arrow)
        self.arrows.clear()
        self.measurements.clear()
        
        # Clear data feed
        self.data_feed.config(state=tk.NORMAL)
        self.data_feed.delete('1.0', tk.END)
        self.data_feed.config(state=tk.DISABLED)
    
    def estimate_distance(self, signal_strength):
        """Estimate distance based on signal strength."""
        reference_strength = -40
        path_loss_exponent = 2
        distance = 10 ** ((reference_strength - signal_strength) / (10 * path_loss_exponent))
        return round(distance, 2)
    
    def update_ui(self):
        """Update UI with real-time data."""
        self.root.after(1000, self.update_ui)

# Run the UI
if __name__ == "__main__":
    root = tk.Tk()
    app = SignalLocalizationUI(root)
    root.mainloop()

import tkinter as tk
from tkinter import Canvas
import numpy as np
import random
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
        
        # Create directional arrow
        self.arrow = self.canvas.create_line(self.center_x, self.center_y, self.center_x, self.center_y - 100, arrow=tk.LAST, width=5, fill="blue")
        
        # Signal strength display
        self.signal_label = tk.Label(root, text="Signal Strength: - dB", font=("Arial", 14))
        self.signal_label.pack(pady=5)
        
        # Distance estimation display
        self.distance_label = tk.Label(root, text="Estimated Distance: - m", font=("Arial", 14))
        self.distance_label.pack(pady=5)
        
        # Reception quality indicator
        self.reception_canvas = Canvas(root, width=200, height=20, bg="white")
        self.reception_canvas.pack(pady=5)
        
        # Live data feed display
        self.data_feed = tk.Text(root, height=6, width=50, state=tk.DISABLED, font=("Arial", 10))
        self.data_feed.pack(pady=5)

        # ML prediction
        self.ml_prediction = (0,0)

        # Longitude and Latitude
        self.longitude = 0
        self.latitude = 0

        # Signal strength
        self.signal_strength = 0

        # Direction
        self.direction = 0

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


    def set_ml_prediction(self, prediction):
        self.ml_prediction = prediction


    def set_signal_strength(self, signal_strength):
        self.signal_strength = signal_strength


    def set_location(self, longitude, latitude):
        self.longitude = longitude
        self.latitude = latitude

    
    def set_direction(self, direction):
        self.direction = direction

    
    def estimate_distance(self, signal_strength):
        """Estimate distance from signal strength using a simple path loss model"""
        reference_strength = -40  # Reference signal strength in dB at 1 meter
        path_loss_exponent = 2  # Typical for indoor environments
        distance = 10 ** ((reference_strength - signal_strength) / (10 * path_loss_exponent))
        return round(distance, 2)
    
    def calculate_angle(self, current_longitude, current_latitude, target_longitude, target_latitude):
        """Calculate angle from current location to target location"""
        d_lon = target_longitude - current_longitude
        d_lat = target_latitude - current_latitude
        angle = np.degrees(np.arctan2(d_lon, d_lat))

        if angle < 0:
            angle += 360
            
        return angle
    
    def update_ui(self):
        """Update the UI with new predictions"""
        predicted_latitude, predicted_longitude = self.ml_prediction
        angle = self.calculate_angle(self.longitude, self.latitude, predicted_longitude, predicted_latitude)
        signal_strength = self.signal_strength
        estimated_distance = self.estimate_distance(signal_strength)
        timestamp = time.strftime("%H:%M:%S")
        
        # Convert angle to new arrow coordinates
        length = 100  # Arrow length
        end_x = self.center_x + length * np.cos(np.radians(-angle + 90))  # Adjusted to match compass correction
        end_y = self.center_y - length * np.sin(np.radians(-angle + 90))
        
        # Update arrow direction
        self.canvas.coords(self.arrow, self.center_x, self.center_y, end_x, end_y)
        
        # Update signal strength and distance labels
        self.signal_label.config(text=f"Signal Strength: {signal_strength} dB")
        self.distance_label.config(text=f"Estimated Distance: {estimated_distance} m")
        
        # Update reception quality indicator
        self.reception_canvas.delete("all")
        quality = max(0, min(100, int((signal_strength + 80) * 2)))  # Scale -80 to -30 dB into 0-100%
        self.reception_canvas.create_rectangle(0, 0, quality * 2, 20, fill="green")
        
        # Update live data feed
        self.data_feed.config(state=tk.NORMAL)
        self.data_feed.insert(tk.END, f"[{timestamp}] Angle: {angle}°, Signal Strength: {signal_strength} dB, Distance: {estimated_distance} m\n")
        self.data_feed.see(tk.END)
        self.data_feed.config(state=tk.DISABLED)
        
        # Repeat update every second
        self.root.after(1000, self.update_ui)

# Run the UI
if __name__ == "__main__":
    root = tk.Tk()
    app = SignalLocalizationUI(root)
    root.mainloop()

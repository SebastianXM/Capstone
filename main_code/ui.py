import tkinter as tk
from tkinter import Canvas
import numpy as np
import math
import time
import threading
from flask import Flask, request
import json

from RxSignal import RxSignal
from utils import calculate_power, estimate_distance
from models import RandomForestModel

compass_direction = 0

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
        self.measure_button = tk.Button(root, text="Measure", command=self.start_signal_reception)
        self.measure_button.pack(pady=5)
        
        self.perform_button = tk.Button(root, text="Perform Algorithm", command=self.perform_algorithm)
        self.perform_button.pack(pady=5)
        
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_measurements)
        self.clear_button.pack(pady=5)
        
        self.ml_prediction = 0
        self.signal_strengths = []
        self.directions = []

        self.flask_thread = threading.Thread(target=run_server, daemon=True)
        self.flask_thread.start()
        
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

    def start_signal_reception(self):
        threading.Thread(target=self.receive_signal, daemon=True).start()

    def receive_signal(self):
        # Receive signal
        print("Receiving signal...")
        rx_file_path = "rx_data.bin"
        rx_signal = RxSignal(output_file=rx_file_path)
        rx_signal.start()
        # Receive signal for 3 seconds
        time.sleep(3) 
        rx_signal.stop()
        rx_signal.wait()
        print("Signal received and saved to file.")

        # Calculate power of received signal
        print("Calculating power...")
        signal_strength = calculate_power(rx_file_path)
        self.signal_strengths.append(signal_strength)

        # Get direction
        print("Getting direction...")
        print("Compass direction:", compass_direction)
        direction = compass_direction
        self.directions.append(direction)

        # Update UI
        estimated_distance = estimate_distance(signal_strength)

        # Draw arrow for this measurement
        length = 100
        end_x = self.center_x + length * np.cos(np.radians(-compass_direction + 90))
        end_y = self.center_y - length * np.sin(np.radians(-compass_direction + 90))
        arrow = self.canvas.create_line(self.center_x, self.center_y, end_x, end_y, arrow=tk.LAST, width=3, fill="red")
        self.arrows.append(arrow)
        
        # Update labels
        self.signal_label.config(text=f"Signal Strength: {signal_strength} dB")
        self.distance_label.config(text=f"Estimated Distance: {estimated_distance} m")
        
        # Log measurement to data feed
        self.data_feed.config(state=tk.NORMAL)
        self.data_feed.insert(tk.END, f"Measured at {compass_direction}°: {signal_strength} dB, Estimated Distance: {estimated_distance} m\n")
        self.data_feed.see(tk.END)
        self.data_feed.config(state=tk.DISABLED)
    
    def perform_algorithm(self):
        self.ml_prediction = RandomForestModel(self.directions, self.signal_strengths) % 360

        # Update the main arrow to point to the predicted angle
        length = 100  # Length of the arrow
        end_x = self.center_x + length * np.cos(np.radians(-self.ml_prediction + 90))
        end_y = self.center_y - length * np.sin(np.radians(-self.ml_prediction + 90))
        
        # Update the main arrow on the canvas
        self.canvas.coords(self.main_arrow, self.center_x, self.center_y, end_x, end_y)
        
        # Log algorithm result to data feed
        self.data_feed.config(state=tk.NORMAL)
        self.data_feed.insert(tk.END, f"Recommended step direction: {self.ml_prediction}°\n")
        self.data_feed.see(tk.END)
        self.data_feed.config(state=tk.DISABLED)
    
    def clear_measurements(self):
        for arrow in self.arrows:
            self.canvas.delete(arrow)
        self.arrows.clear()
        
        # Clear data feed
        self.data_feed.config(state=tk.NORMAL)
        self.data_feed.delete('1.0', tk.END)
        self.data_feed.config(state=tk.DISABLED)
    
    def update_ui(self):
        """Update UI with real-time data."""
        self.root.after(1000, self.update_ui)

def create_app():
    app = Flask(__name__)

    @app.route('/data', methods=['POST'])
    def receive_data():
        global compass_direction
        try:
            data = request.get_json(force=True)
            payload = data.get('payload')

            for entry in payload:
                values = entry.get('values', {})
                if 'magneticBearing' in values:
                    compass_direction = values.get('magneticBearing')

            print("Flask Server Received direction:", compass_direction)
            return "200"
        except Exception as e:
            print("Error in receive_data:", e)
            return "400"

    return app

def run_server():
    app = create_app()
    app.run(host="0.0.0.0", port=8002, debug=False, use_reloader=False)

# Run the UI
if __name__ == "__main__":
    root = tk.Tk()
    app = SignalLocalizationUI(root)
    root.mainloop()

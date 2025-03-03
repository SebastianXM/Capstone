import tkinter as tk
from tkinter import Canvas
import numpy as np
import math
import time
import random

class SignalLocalizationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Localization UI")
        
        self.canvas = Canvas(root, width=400, height=400, bg="white")
        self.canvas.pack()

        self.draw_grid()
        
        self.center_x, self.center_y = 200, 200
        self.radius = 120
        self.draw_compass()
        
        self.arrow = self.canvas.create_line(self.center_x, self.center_y, 
                                             self.center_x, self.center_y - 100, 
                                             arrow=tk.LAST, width=5, fill="blue")
        self.current_angle = 0
        
        self.signal_label = tk.Label(root, text="Signal Strength: - dB", font=("Arial", 14))
        self.signal_label.pack(pady=5)

        self.distance_label = tk.Label(root, text="Estimated Distance: - m", font=("Arial", 14))
        self.distance_label.pack(pady=5)

        self.reception_canvas = Canvas(root, width=200, height=20, bg="white")
        self.reception_canvas.pack(pady=5)
        
        self.data_feed = tk.Text(root, height=6, width=50, state=tk.DISABLED, font=("Arial", 10))
        self.data_feed.pack(pady=5)

        self.ml_prediction = (0,0)
        self.longitude = 0
        self.latitude = 0
        self.signal_strength = 0
        self.direction = 0
        
        self.update_ui()
    
    def draw_grid(self):
        grid_spacing = 40
        for x in range(0, 400, grid_spacing):
            self.canvas.create_line(x, 0, x, 400, fill="lightgray")
        for y in range(0, 400, grid_spacing):
            self.canvas.create_line(0, y, 400, y, fill="lightgray")
    
    def draw_compass(self):
        directions = {0: "N", 90: "E", 180: "S", 270: "W"}
        for angle in range(0, 360, 30):
            radian = np.radians(-angle + 90)
            x1 = self.center_x + (self.radius - 10) * np.cos(radian)
            y1 = self.center_y - (self.radius - 10) * np.sin(radian)
            x2 = self.center_x + self.radius * np.cos(radian)
            y2 = self.center_y - self.radius * np.sin(radian)
            self.canvas.create_line(x1, y1, x2, y2, fill="black")

            text_x = self.center_x + (self.radius + 20) * np.cos(radian)
            text_y = self.center_y - (self.radius + 20) * np.sin(radian)
            label_text = directions.get(angle, str(angle))
            self.canvas.create_text(text_x, text_y, text=label_text, font=("Arial", 12, "bold"), fill="black")
    
    def vincenty_initial_bearing(self, lat1, lon1, lat2, lon2):
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        lambda1, lambda2 = math.radians(lon1), math.radians(lon2)
        delta_lambda = lambda2 - lambda1

        x = math.sin(delta_lambda) * math.cos(phi2)
        y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)

        theta = math.atan2(x, y)
        return (math.degrees(theta) + 360) % 360  

    def animate_arrow(self, target_angle, steps=10, step_delay=50):
        start_angle = self.current_angle
        angle_diff = (target_angle - start_angle) / steps

        def step_animation(step=0):
            if step >= steps:
                self.current_angle = target_angle
                return
            
            new_angle = start_angle + angle_diff * step
            length = 100
            end_x = self.center_x + length * np.cos(np.radians(-new_angle + 90))
            end_y = self.center_y - length * np.sin(np.radians(-new_angle + 90))
            self.canvas.coords(self.arrow, self.center_x, self.center_y, end_x, end_y)

            self.root.after(step_delay, step_animation, step + 1)

        step_animation()

    def update_ui(self):
        predicted_latitude, predicted_longitude = self.ml_prediction
        bearing = self.vincenty_initial_bearing(self.latitude, self.longitude, predicted_latitude, predicted_longitude)
        self.animate_arrow(bearing)
        
        signal_strength = self.signal_strength
        estimated_distance = round(10 ** ((-40 - signal_strength) / 20), 2)
        
        self.signal_label.config(text=f"Signal Strength: {signal_strength} dB")
        self.distance_label.config(text=f"Estimated Distance: {estimated_distance} m")
        
        quality = max(0, min(100, int((signal_strength + 80) * 2)))
        self.reception_canvas.delete("all")
        self.reception_canvas.create_rectangle(0, 0, quality * 2, 20, fill="green")
        
        self.data_feed.config(state=tk.NORMAL)
        self.data_feed.insert(tk.END, f"Bearing: {bearing:.2f}°, Signal: {signal_strength} dB, Distance: {estimated_distance} m\n")
        self.data_feed.see(tk.END)
        self.data_feed.config(state=tk.DISABLED)
        
        self.root.after(1000, self.update_ui)

if __name__ == "__main__":
    root = tk.Tk()
    app = SignalLocalizationUI(root)
    root.mainloop()
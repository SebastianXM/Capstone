import tkinter as tk
from tkinter import Canvas, ttk
from ttkthemes import ThemedTk

import numpy as np
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SignalLocalizationUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Localization UI (Arc Theme)")

        # ==================================================
        #  TOP & BOTTOM FRAMES
        # ==================================================
        self.top_frame = ttk.Frame(root, padding=10)
        self.top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.bottom_frame = ttk.Frame(root, padding=10)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # ==================================================
        #  LEFT: COMPASS CANVAS
        # ==================================================
        self.left_frame = ttk.Frame(self.top_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = Canvas(
            self.left_frame, 
            width=400, 
            height=400, 
            bg="white"
        )
        self.canvas.pack(side=tk.TOP, padx=5, pady=5)

        # Draw the compass grid and markings
        self.draw_grid()
        self.center_x, self.center_y = 200, 200
        self.radius = 120
        self.draw_compass()

        # Main directional arrow
        self.arrows = []
        self.main_arrow = self.canvas.create_line(
            self.center_x, self.center_y,
            self.center_x, self.center_y - 100,
            arrow=tk.LAST, width=5, fill="blue"
        )
        self.current_angle = 0

        # ==================================================
        #  RIGHT: MEASUREMENTS, DATA, BUTTONS
        # ==================================================
        self.right_frame = ttk.Frame(self.top_frame)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # -- Measurement info at the top --
        self.measurement_frame = ttk.LabelFrame(
            self.right_frame, text="Measurements", padding=10
        )
        self.measurement_frame.pack(side=tk.TOP, fill=tk.X)

        self.signal_label = ttk.Label(self.measurement_frame, text="Signal Strength: - dB")
        self.signal_label.pack(pady=(0, 5))

        self.distance_label = ttk.Label(self.measurement_frame, text="Estimated Distance: - m")
        self.distance_label.pack(pady=(0, 5))

        # -- Data feed text box in the middle --
        self.data_feed = tk.Text(self.right_frame, height=6, bg="#ffffff")
        self.data_feed.config(state=tk.DISABLED)
        self.data_feed.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)

        # -- Buttons at the bottom, uniform width --
        self.button_frame = ttk.Frame(self.right_frame)
        self.button_frame.pack(side=tk.BOTTOM, pady=10)

        button_width = 15
        self.measure_button = ttk.Button(
            self.button_frame, text="Measure",
            command=self.measure_direction,
            width=button_width
        )
        self.measure_button.pack(pady=5)

        self.perform_button = ttk.Button(
            self.button_frame, text="Perform Algorithm",
            command=self.perform_algorithm,
            width=button_width
        )
        self.perform_button.pack(pady=5)

        self.clear_button = ttk.Button(
            self.button_frame, text="Clear",
            command=self.clear_measurements,
            width=button_width
        )
        self.clear_button.pack(pady=5)

        # ==================================================
        #  MEASUREMENT / SIGNAL VARIABLES
        # ==================================================
        self.measurements = []
        self.signal_history = []
        self.signal_strength = 0
        self.longitude = 0
        self.latitude = 0

        # ==================================================
        #  MATPLOTLIB GRAPH: WHITE BACKGROUND + GRID LINES
        # ==================================================
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update({
            "figure.facecolor": "#f0f0f0",  # around the plot area
            "axes.facecolor": "#ffffff",    # white area for the plot
            "axes.edgecolor": "black",
            "font.family": "Helvetica",
            "font.size": 11,
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        })

        # Create figure with higher DPI for clarity
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=120)
        self.ax.set_title("Signal Strength Over Time")
        self.ax.set_xlabel("Measurement Count")
        self.ax.set_ylabel("Signal Strength (dB)")

        # Add solid grid lines to match compass
        self.ax.grid(True, linestyle='-', color='lightgray')

        # A line for dynamic data
        self.line, = self.ax.plot([], [], marker='o', linestyle='-')

        self.fig.tight_layout()
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=self.bottom_frame)
        self.canvas_graph.draw()
        self.canvas_graph.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Schedule periodic updates
        self.update_ui()

    # ======================================================
    #                   EVENT HANDLERS
    # ======================================================
    def measure_direction(self):
        angle = self.current_angle
        signal_strength = self.signal_strength
        estimated_distance = self.estimate_distance(signal_strength)

        self.measurements.append((angle, signal_strength))
        self.signal_history.append(signal_strength)

        # Draw an arrow for the measurement
        length = 100
        end_x = self.center_x + length * np.cos(np.radians(-angle + 90))
        end_y = self.center_y - length * np.sin(np.radians(-angle + 90))
        arrow = self.canvas.create_line(
            self.center_x, self.center_y,
            end_x, end_y,
            arrow=tk.LAST, width=3, fill="red"
        )
        self.arrows.append(arrow)

        # Update graph data
        self.line.set_xdata(range(len(self.signal_history)))
        self.line.set_ydata(self.signal_history)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas_graph.draw()

        # Update labels
        self.signal_label.config(text=f"Signal Strength: {signal_strength} dB")
        self.distance_label.config(text=f"Estimated Distance: {estimated_distance} m")

        # Log measurement
        self.data_feed.config(state=tk.NORMAL)
        self.data_feed.insert(
            tk.END,
            f"Measured at {angle}°: {signal_strength} dB, Distance: {estimated_distance} m\n"
        )
        self.data_feed.see(tk.END)
        self.data_feed.config(state=tk.DISABLED)

    def perform_algorithm(self):
        if len(self.measurements) >= 3:
            # Simple example: average angle as "best direction"
            best_direction = sum(angle for angle, _ in self.measurements) // len(self.measurements)
            self.animate_arrow(best_direction)

            self.data_feed.config(state=tk.NORMAL)
            self.data_feed.insert(tk.END, f"Recommended step direction: {best_direction}°\n")
            self.data_feed.see(tk.END)
            self.data_feed.config(state=tk.DISABLED)

    def clear_measurements(self):
        # Clear the arrows
        for arrow in self.arrows:
            self.canvas.delete(arrow)
        self.arrows.clear()
        self.measurements.clear()

        # Clear data feed
        self.data_feed.config(state=tk.NORMAL)
        self.data_feed.delete('1.0', tk.END)
        self.data_feed.config(state=tk.DISABLED)

        # Clear the plot
        self.line.set_xdata([])
        self.line.set_ydata([])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas_graph.draw()

    def animate_arrow(self, target_angle):
        length = 100
        end_x = self.center_x + length * np.cos(np.radians(-target_angle + 90))
        end_y = self.center_y - length * np.sin(np.radians(-target_angle + 90))
        self.canvas.coords(self.main_arrow, self.center_x, self.center_y, end_x, end_y)
        self.current_angle = target_angle

    # ======================================================
    #                 DRAWING METHODS
    # ======================================================
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
            self.canvas.create_text(
                text_x, text_y, text=label_text,
                font=("Arial", 12, "bold"), fill="black"
            )

    def estimate_distance(self, signal_strength):
        reference_strength = -40
        path_loss_exponent = 2
        distance = 10 ** ((reference_strength - signal_strength) / (10 * path_loss_exponent))
        return round(distance, 2)

    # ======================================================
    #              PERIODIC UPDATES
    # ======================================================
    def update_ui(self):
        # For real-time, you'd update self.signal_strength or angle here
        self.root.after(1000, self.update_ui)


if __name__ == "__main__":
    root = ThemedTk(theme="arc")
    app = SignalLocalizationUI(root)
    root.mainloop()

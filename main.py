import os
import time
import torch
import pickle
import sklearn
import torch.nn as nn
import math
import numpy as np
from gnuradio import gr, blocks, uhd
import threading
import queue
import tkinter as tk
from tkinter import Tk
from flask import Flask, request
import json

from signallocalizationui import SignalLocalizationUI

last_values = {'x': 0, 'y': 0, 'z': 0}

class RegressionNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(RegressionNN, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, hidden_size)
    self.fc4 = nn.Linear(hidden_size, hidden_size)
    self.fc5 = nn.Linear(hidden_size, output_size)


  def forward(self, x):
    out = self.fc1(x)
    out = self.relu(out)
    out = self.fc2(out)
    out = self.relu(out)
    out = self.fc3(out)
    out = self.relu(out)
    out = self.fc4(out)
    out = self.relu(out)
    out = self.fc5(out)
    return out
  

class RxSignal(gr.top_block):
    def __init__(self, samp_rate=64000, output_file="rx_data.bin"):
        gr.top_block.__init__(self, "Signal Receiver")
        self.samp_rate = samp_rate
        self.output_file = output_file

        # USRP Source
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(("serial=3080C7F", "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0, 1)),
            ),
        )
        self.uhd_usrp_source_0.set_center_freq(2400000000, 0)
        self.uhd_usrp_source_0.set_gain(40, 0)
        self.uhd_usrp_source_0.set_antenna('RX2', 0)
        self.uhd_usrp_source_0.set_samp_rate(2e6)
        self.uhd_usrp_source_0.set_bandwidth(2e6, 0)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec())

        # File Sink
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex * 1, output_file, False)
        self.blocks_file_sink_0.set_unbuffered(False)

        # Connections
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_file_sink_0, 0))

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)


# Function to calculate power
def calculate_power(rx_file_path):
    with open(rx_file_path, 'rb') as f:
        rx_data = np.fromfile(f, dtype=np.complex64)

    rx_power = 10 * np.log10(np.mean(np.abs(rx_data)**2))

    print("Rx Power: ", rx_power)
    return rx_power


def create_flask_app():
    app = Flask(__name__)

    @app.route('/data', methods=['POST'])
    def recieve_data():
        global last_values
        try:
            data = json.loads(request.data)
            payload = data.get('payload')

            # Extract data from payload
            for entry in payload:
                values = entry.get('values')
                x = values.get('x')
                y = values.get('y')
                z = values.get('z')

                last_values = {'x': x, 'y': y, 'z': z}
                print(f"X: {x}, Y: {y}, Z: {z}")

            return "200"
        except Exception as e:
            print(e)
            return "400"
    return app

# Function to get direction
def get_direction(host, port, duration=5):
    global last_values

    app = create_flask_app()
    
    server_thread = threading.Thread(target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False))
    server_thread.start()

    try:
        print(f"Flask server started at {host}:{port}")
        threading.Event().wait(duration)
    finally:
        print("Shutting down Flask server...")
        server_thread.join()
    
    return last_values['x'], last_values['y'], last_values['z']


def estimate_distance(rx_power):
    """Estimate distance from signal strength using a simple path loss model"""
    reference_strength = -40
    path_loss_exponent = 2
    distance = 10 ** ((reference_strength - rx_power) / (10 * path_loss_exponent))
    return round(distance, 9)


def background_task(update_queue):
    while True:
        # Receive signal for 5 seconds
        print("Receiving signal...")
        rx_file_path = "rx_data.bin"
        rx_signal = RxSignal(output_file=rx_file_path)
        rx_signal.start()
        # Receive signal for 5 seconds
        time.sleep(5) 
        rx_signal.stop()
        rx_signal.wait()
        print("Signal received and saved to file.")

        # Calculate power of received signal
        print("Calculating power...")
        rx_power = calculate_power(rx_file_path)

        # Get direction
        print("Getting direction...")
        host = "10.61.3.198"
        port = 8000
        x_direction, y_direction, z_direction = get_direction(host, port, duration=5)

        # Input data to the ML model
        print("Inputting data to the ML model...")
        if x_direction and y_direction and z_direction:
            print(f"Last known direction: x: {x_direction}, y: {y_direction}, z: {z_direction}\n")
            print(f"Rx Power: {rx_power}")
        '''
            try:
                # NN model
                model = RegressionNN(5, 64, 2)
                model.load_state_dict(torch.load("NN_model.pth",weights_only=True))
                model.eval()
                with torch.no_grad():
                    input_data = torch.tensor([rx_longitude, rx_latitude, longitude_delta, latitude_delta, rx_power]).float()
                    output = model(input_data)
                    predicted_latitude = output[0]
                    predicted_longitude = output[1]
                    print(f"Predicted location: Latitude={predicted_latitude}, Longitude={predicted_longitude}")
                
                # ML model
                ml_model = pickle.load(open("ml_model.pkl", "rb"))
                output = ml_model.predict([[rx_longitude, rx_latitude, longitude_delta, latitude_delta, rx_power, estimated_distance]])
                predicted_longitude = output[0][0]
                predicted_latitude = output[0][1]
                predicted_distance = output[0][2]
                estimated_angle = estimate_angle(rx_longitude, rx_latitude, predicted_longitude, predicted_latitude)

                new_longitude, new_latitude = add_distance_to_long_lat(rx_longitude, rx_latitude, predicted_distance, estimated_angle)
                print(f"Prediction: Latitude={new_latitude}, Longitude={new_longitude}, Distance={predicted_distance} meters, Angle={estimated_angle} degrees")

            except Exception as e:
                print(f"Error: {e}")
                predicted_latitude, predicted_longitude = 0, 0
        else:
            predicted_latitude, predicted_longitude = 0, 0
        '''
        updates = {
            "signal_strength": rx_power,
            "direction": (x_direction, y_direction, z_direction),
            "ml_prediction": (0)
        }
        update_queue.put(updates)
        print("Updates sent to UI\n")
        time.sleep(5)


def process_queue(ui, update_queue):
    try:
        while not update_queue.empty():
            updates = update_queue.get_nowait()
            ui.set_ml_prediction(updates["ml_prediction"])
            ui.set_signal_strength(updates["signal_strength"])
            ui.set_direction(updates["direction"])
            ui.update_ui()
    except queue.Empty:
        pass
    ui.root.after(1000, process_queue, ui, update_queue)


def main():
    update_queue = queue.Queue()

    bg_thread = threading.Thread(target=background_task, args=(update_queue,), daemon=True)
    bg_thread.start()

    root = Tk()
    ui = SignalLocalizationUI(root)

    root.after(1000, process_queue, ui, update_queue)

    root.mainloop()


if __name__ == "__main__":
    main()

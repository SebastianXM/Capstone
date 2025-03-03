import os
import time
import torch
import pickle
import sklearn
import torch.nn as nn
import socket
import math
import pynmea2
import numpy as np
from gnuradio import gr, blocks, uhd
import threading
import queue
import tkinter as tk
from tkinter import Tk

from signallocalizationui import SignalLocalizationUI


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


# Function to get location and direction
def get_location_and_direction(host, port, duration=5):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Socket created successfully")
    except socket.error as err:
        print(f"Socket creation failed with error: {err}")
        return None

    try:
        client_socket.connect((host, port))
        print("Connected to the server successfully")
    except socket.error as err:
        print(f"Connection failed with error: {err}")
        client_socket.close()
        return None

    start_time = time.time()
    location_data = []
    direction_data = []

    while time.time() - start_time < duration:
        try:
            data = client_socket.recv(1024).decode("utf-8")
            if data:
                sentences = data.split("\r\n")
                for sentence in sentences:
                    if sentence.strip():
                        try:
                            msg = pynmea2.parse(data)
                            
                            if isinstance(msg, pynmea2.types.RMC):
                                location_data.append((msg.latitude, msg.longitude))
                            elif isinstance(msg, pynmea2.types.HDT):
                                direction_data.append(msg.heading)
                                print(f"Direction: {msg.heading}")
                            else:
                                print(f"Unhandled NMEA message type")
                        except pynmea2.ParseError as parse_err:
                            print(f"Parse error: {parse_err}")
            else:
                print("No data received; the server may have closed the connection.")
                break
        except socket.error as err:
            print(f"Data receive failed with error: {err}")
            break

    client_socket.close()
    return location_data, direction_data


def calculate_lat_long_direction(direction):
    direction_rad = math.radians(direction)

    delta_lat = math.cos(direction_rad)
    delta_long = math.sin(direction_rad)

    return delta_lat, delta_long


def estimate_distance(rx_power):
    """Estimate distance from signal strength using a simple path loss model"""
    reference_strength = -40
    path_loss_exponent = 2
    distance = 10 ** ((reference_strength - rx_power) / (10 * path_loss_exponent))
    return round(distance, 9)


def estimate_angle(current_longitude, current_latitude, target_longitude, target_latitude):
    """Calculate angle from current location to target location"""
    d_lon = target_longitude - current_longitude
    d_lat = target_latitude - current_latitude
    angle = np.degrees(np.arctan2(d_lon, d_lat))

    if angle < 0:
        angle += 360

    return angle


def add_distance_to_long_lat(current_longitude, current_latitude, distance, angle):
    earth_radius = 6371000
    
    lat_rad = math.radians(current_latitude)
    lon_rad = math.radians(current_longitude)
    angle_rad = math.radians(angle)

    new_lat = math.asin(math.sin(lat_rad) * math.cos(distance / earth_radius) + math.cos(lat_rad) * math.sin(distance / earth_radius) * math.cos(angle_rad))
    new_lon = lon_rad + math.atan2(math.sin(angle_rad) * math.sin(distance / earth_radius) * math.cos(lat_rad), math.cos(distance / earth_radius) - math.sin(lat_rad) * math.sin(new_lat))

    new_lat = math.degrees(new_lat)
    new_lon = math.degrees(new_lon)

    return new_lon, new_lat


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

        # Get location and direction
        print("Getting location and direction...")
        host = "10.61.2.141"
        port = 11123
        location_data, direction_data = get_location_and_direction(host, port, duration=5)

        if location_data and direction_data:
            print("Location and direction data received")
            last_location = location_data[-1]
            rx_latitude = round(last_location[0], 9)
            rx_longitude = round(last_location[1], 9)
            rx_direction = direction_data[-1]
        else:
            print("Failed to get location or direction data.")
            rx_latitude, rx_longitude, rx_direction = 0, 0, 0

        # Input data to the ML model
        print("Inputting data to the ML model...")
        if location_data and direction_data:

            print(f"Last known location and direction: Latitude={rx_latitude}, Longitude={rx_longitude}, Direction={rx_direction}\n")
            print(f"Rx Power: {rx_power}")

            latitude_delta, longitude_delta = calculate_lat_long_direction(rx_direction)
            print(f"Latitude delta: {latitude_delta}, Longitude delta: {longitude_delta}")
            estimated_distance = estimate_distance(rx_power)
            try:
                # NN model
                '''
                model = RegressionNN(5, 64, 2)
                model.load_state_dict(torch.load("NN_model.pth",weights_only=True))
                model.eval()
                with torch.no_grad():
                    input_data = torch.tensor([rx_longitude, rx_latitude, longitude_delta, latitude_delta, rx_power]).float()
                    output = model(input_data)
                    predicted_latitude = output[0]
                    predicted_longitude = output[1]
                    print(f"Predicted location: Latitude={predicted_latitude}, Longitude={predicted_longitude}")
                '''
                
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
            
        updates = {
            "ml_prediction": (float(new_latitude), float(new_longitude)),
            "signal_strength": rx_power,
            "location": (rx_longitude, rx_latitude),
            "direction": rx_direction
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
            ui.set_location(*updates["location"])
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

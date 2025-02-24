import os
import time
import torch
import sklearn
import torch.nn as nn
import socket
import pynmea2
import numpy as np
from gnuradio import gr, blocks, uhd
import pickle


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
        self.uhd_usrp_source_0.set_center_freq(2.402e9, 0)
        self.uhd_usrp_source_0.set_gain(0, 0)
        self.uhd_usrp_source_0.set_antenna('RX2', 0)
        self.uhd_usrp_source_0.set_samp_rate(2e6)
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

    while time.time() - start_time < duration:
        try:
            data = client_socket.recv(1024).decode("utf-8")
            if data:
                try:
                    msg = pynmea2.parse(data)
                    location_data.append((msg.latitude, msg.longitude))
                except pynmea2.ParseError as parse_err:
                    print(f"Parse error: {parse_err}")
            else:
                print("No data received; the server may have closed the connection.")
                break
        except socket.error as err:
            print(f"Data receive failed with error: {err}")
            break

    client_socket.close()
    return location_data


# Main program loop
def main():
    print("Starting the program...")

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
    host = "192.168.1.166"
    port = 11123
    location_data = get_location_and_direction(host, port, duration=5)
    if location_data:
        print("Location data received")
    else:
        print("Failed to get location data.")

    # Input data to the ML model
    print("Inputting data to the ML model...")
    if location_data:
        last_location = location_data[-1]
        rx_latitude = round(last_location[0], 9)
        rx_longitude = round(last_location[1], 9)
        print(f"Last known location: Latitude={rx_latitude}, Longitude={rx_longitude}\n")
        print(f"Rx Power: {rx_power}")
        
        # NN model
        '''
        model = RegressionNN(5, 64, 2)
        model.load_state_dict(torch.load("NN_model.pth",weights_only=True))
        model.eval()
        with torch.no_grad():
            input_data = torch.tensor([rx_longitude, rx_latitude, 0, 0, rx_power]).float()
            output = model(input_data)
            print(f"Predicted location: Latitude={output[0]}, Longitude={output[1]}")
        '''

        # ML model
        ml_model = pickle.load(open("ml_model.pkl", "rb"))
        output = ml_model.predict([[rx_longitude, rx_latitude, 0, 0, rx_power]])
        print(f"Predicted location: Latitude={output[0][0]}, Longitude={output[0][1]}")
        
    print("Program completed.")


if __name__ == "__main__":
    main()

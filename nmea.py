import time
import socket
import pynmea2

host = "10.61.2.141"
port = 11123

# Create a socket object
try: 
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Socket created successfully")
except socket.error as err:
    print(f"Socket creation failed with error: {err}")
    exit(1)

# Connect to the server
try:
    client_socket.connect((host, port))
    print("Connected to the server successfully")
except socket.error as err:
    print(f"Connection failed with error: {err}")
    client_socket.close()
    exit(1)

start_time = time.time()
duration = 60 

# Receive data from the server for 30 seconds
while time.time() - start_time < duration:
    try:
        # Receive data from the server
        data = client_socket.recv(1024).decode("utf-8")
        print(data) 

        if data:
            # Split the data into individual NMEA sentences
            sentences = data.split("\r\n")  # NMEA sentences are separated by \r\n
            for sentence in sentences:
                if sentence.strip():  # Ignore empty lines
                    try:
                        # Parse the individual NMEA sentence
                        msg = pynmea2.parse(sentence)
                        print(msg)

                        # Handle specific NMEA message types
                        if isinstance(msg, pynmea2.types.RMC):  # Recommended Minimum Navigation Information
                            print(f"Latitude: {msg.latitude}, Longitude: {msg.longitude}, Course: {msg.true_course}")
                        elif isinstance(msg, pynmea2.types.GGA):  # Global Positioning System Fix Data
                            print(f"Latitude: {msg.latitude}, Longitude: {msg.longitude}, Altitude: {msg.altitude} {msg.altitude_units}")
                        elif isinstance(msg, pynmea2.types.HDT):  # Heading, True
                            print(f"Heading (True): {msg.heading}")
                        elif isinstance(msg, pynmea2.types.HDM):
                            print(f"Heading (Magnetic): {msg.heading}")
                        elif isinstance(msg, pynmea2.types.HDG):
                            print(f": {msg.heading}")
                        else:
                            print(f"Unhandled NMEA message type: {msg.sentence_type}")

                    except pynmea2.ParseError as parse_err:
                        print(f"Parse error: {parse_err}")
        else:
            print("No data received; the server may have closed the connection.")
            break
    except socket.error as err:
        print(f"Data receive failed with error: {err}")
        break

client_socket.close()

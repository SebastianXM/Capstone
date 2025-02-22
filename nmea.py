import time
import socket
import pynmea2

host = "192.168.1.166"
port = 11123

# create a socket object
try: 
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Socket created successfully")
except socket.error as err:
    print(f"Socket creation failed with error: {err}")
    exit(1)

# connect to the server
try:
    client_socket.connect((host, port))
    print("Connected to the server successfully")
except socket.error as err:
    print(f"Connection failed with error: {err}")
    client_socket.close()
    exit(1)

start_time = time.time()
duration = 30 

# receive data from the server for 30 seconds
while time.time() - start_time < duration:
    try:
        data = client_socket.recv(1024).decode("utf-8")
        if data:
            try:
                msg = pynmea2.parse(data)
                print(f"Latitude: {msg.latitude}, Longitude: {msg.longitude}")
            except pynmea2.ParseError as parse_err:
                print(f"Parse error: {parse_err}")
        else:
            print("No data received; the server may have closed the connection.")
            break
    except socket.error as err:
        print(f"Data receive failed with error: {err}")
        break

client_socket.close()


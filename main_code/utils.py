import numpy as np

# Function to calculate power from file
def calculate_power(rx_file_path):
    with open(rx_file_path, 'rb') as f:
        rx_data = np.fromfile(f, dtype=np.complex64)
    rx_power = 10 * np.log10(np.mean(np.abs(rx_data)**2))
    print("Rx Power: ", round(rx_power, 3))
    return round(rx_power, 3)

def estimate_distance(signal_strength):
    reference_strength = -40
    path_loss_exponent = 2
    distance = 10 ** ((reference_strength - signal_strength) / (10 * path_loss_exponent))
    return round(distance, 2)
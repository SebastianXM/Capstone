import numpy as np

tx_file_path="tx_cosine_data.bin"
rx_file_path="rx_cosine_data.bin"

with open(tx_file_path, 'rb') as f:
    tx_data = np.fromfile(f, dtype=np.complex64)

with open(rx_file_path, 'rb') as f:
    rx_data = np.fromfile(f, dtype=np.complex64)

tx_power = 20 * np.log10(np.mean(np.abs(tx_data)**2))
rx_power = 20 * np.log10(np.mean(np.abs(rx_data)**2))


print("Tx Power: ", tx_power)
print("Rx Power: ", rx_power)

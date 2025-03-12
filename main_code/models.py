import torch
import pickle
import sklearn
import torch.nn as nn
import math
import pandas as pd

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

def get_azimuthal_angle(compass_angle):
    return compass_angle % 360

def calculate_heuristic_azimuth(azimuths):
    # Convert azimuths to unit vectors in the complex plane
    unit_vectors = [complex(math.cos(math.radians(a)), math.sin(math.radians(a))) for a in azimuths]
    # Compute the average vector
    avg_vector = sum(unit_vectors) / len(unit_vectors)
    # Convert back to an angle in degrees
    heuristic_azimuth = math.degrees(math.atan2(avg_vector.imag, avg_vector.real))
    # Normalize to [0, 360)
    return heuristic_azimuth % 360

def RandomForestModel(compass_angles, powers):
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)

    azimuth_1, azimuth_2, azimuth_3 = get_azimuthal_angle(compass_angles[-1]), get_azimuthal_angle(compass_angles[-2]), get_azimuthal_angle(compass_angles[-3])
    power_1, power_2, power_3 = powers[-1], powers[-2], powers[-3]
    heurstic_azimuth = calculate_heuristic_azimuth([azimuth_1, azimuth_2, azimuth_3])

    values = [azimuth_1, azimuth_2, azimuth_3, power_1, power_2, power_3, heurstic_azimuth]
    feature_names = ['Azimuth_1c', 'Azimuth_2c', 'Azimuth_3c', 'Power_1', 'Power_2', 'Power_3','Heuristic_Azc']
    features = pd.DataFrame([values], columns=feature_names)

    prediction = model.predict(features)
    print(f"Predicted direction: {prediction[0] % 360}°")
    return prediction[0]

    

    
    
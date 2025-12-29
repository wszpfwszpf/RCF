from utils.io.psee_loader import PSEELoader
from tqdm import tqdm
import numpy as np, matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams['font.family'] = 'Times New Roman'
file_path = r'C:\Users\93084\Desktop\自己论文写作\2.降噪\RCF\data\DV\Off_set1_trail1.dat'   # The file path of the dynamic vision data (.dat format)

event_time = 4.8e6  # Select the time duration of dynamic vision data to be extracted
video = PSEELoader(file_path)  # Read the dynamic vision data file
events = video.load_delta_t(event_time)  # Extract data within the selected time window event_time
events['t'].sort()  # Sort event data by timestamp
Oral_event_samples_clip = [event for event in tqdm(events, desc='Time interception') if 3.1e6 <= event['t'] <=4.8e6]  # Further extract the valid event stream

# Visualize events as a 2D image
Img_pos = np.zeros(shape=(720, 1280))  # Ensure that the canvas range does not exceed 720 (height) × 1280 (width)
for event in Oral_event_samples_clip:
    if event['p'] == 1:
        Img_pos[event['y'], event['x']] = 1

plt.figure()
plt.imshow(Img_pos)
plt.show()





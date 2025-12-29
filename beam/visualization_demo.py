from utils.io.psee_loader import PSEELoader
from tqdm import tqdm
import numpy as np, matplotlib.pyplot as plt
from matplotlib import rcParams
from utils.io.scatter_plot import plot_event_2d

rcParams['font.family'] = 'Times New Roman'
file_path = r'C:\Users\93084\Desktop\自己论文写作\2.降噪\RCF\data\DV\Off_set1_trail1.dat'  # The file path of the dynamic vision data (.dat format)

event_time = 4.8e6  # Select the time duration of dynamic vision data to be extracted
video = PSEELoader(file_path)  # Read the dynamic vision data file
events = video.load_delta_t(event_time)  # Extract data within the selected time window event_time
events['t'].sort()  # Sort event data by timestamp
Oral_event_samples_clip = [event for event in tqdm(events, desc='Time interception') if 3.1e6 <= event['t'] <=4.8e6]  # Further extract the valid event stream

# Example: visualize events within the ROI defined by a common y-coordinate of 50
# and an x-coordinate range from 600 to 670.
# The y-coordinate and x-range can be adjusted as needed.
Oral_event_samples_clip_y = [event for event in tqdm(Oral_event_samples_clip) if 600 <= event[1] <= 670] #  Select the x-range of the ROI for visualization
plot_event_2d(Oral_event_samples_clip_y, y_aix=50)   # Choose a specific y-coordinate from the ROI

#------------------------ To visualize along the x-axis instead, modify the selection as follows:--------------------------------------------------------------#
# Oral_event_samples_clip_y = [event for event in tqdm(Oral_event_samples_clip) if 600 <= event[2] <= 670] #  Select the y-range of the ROI for visualization
# plot_event_2d(Oral_event_samples_clip_y, x_aix=50)   # Choose a specific x-coordinate from the ROI

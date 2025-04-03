# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:23:37 2025

@author: mnl5205
"""
import argparse
import cv2
import pandas as pd
import os
import time

# Program to allow the user (hand coder) to manually label each video file, to be compared with the automatic labeling program in the bffs_v8 file.

# Retrieves information from user/manual coder
parser = argparse.ArgumentParser()
# -sub SUBJECT -vid VIDEO (Maybe omit this line for clarity, unless it is necessary for potential future changes?)
parser.add_argument("-s", "--subject", dest="subject", help="Subject Number")
parser.add_argument("-t", "--timepoint", dest="timepoint", help="Timepoint")
parser.add_argument("-u", "--user", dest="user", help="psu ID")

args = parser.parse_args()
timepoint = args.timepoint
user = args.user
subject = args.subject

# Based on information, identify location of video file and output from YOLO pipeline
# Specifies location of input file based on user input 'subject'
folder = os.path.join('C:/Users',user,'OneDrive - The Pennsylvania State University/PCAT R01/data/data_processing',timepoint,'eyetracker_data/tsst/processed',
    subject
)
video_path =  os.path.join(folder, subject + '_hand_out.avi') # file path for the 1 minute video, for the user to manually label
gaze_path = os.path.join(folder, subject + '_met_data.csv') # file path for output from YOLO pipeline, to compare to the manual labeling

# Specifies location of output DataFrame
output_folder = os.path.join('C:/Users',user,'OneDrive - The Pennsylvania State University/PCAT R01/data/data_processing',timepoint,'eyetracker_data/tsst/validation'
)
output_csv = os.path.join(output_folder, subject + '_validation_output_reliability.csv')

# Read in our data output from YOLO pipeline
gaze_data = pd.read_csv(gaze_path)

# Filter to only include validation frames
gaze_data = gaze_data.loc[gaze_data['Validation'] == 1]
gaze_data = gaze_data.reset_index(drop=True)

input_data = []  # List to store the input numbers for each frame

# Open the video file
cap = cv2.VideoCapture(video_path)

# Create a starting time so we can track how long it takes for the user to manually label
st = time.time()

# Function to handle potential video file errors
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0

# Function to make sure user doesn't input a typo / code that doesn't exist
def get_valid_number():
    valid_values = {0, 1, 2, 3, 4, 99}
    while True:
        try:
            user_input = input("Enter a number (0, 1, 2, 3, 4, or 99): ")
            num = int(user_input)
            if num in valid_values:
                return num
            else:
                print("Invalid input. Please enter one of the following: 0, 1, 2, 3, 4, or 99.")
        except ValueError:
            print("Invalid input. Please enter a number (0, 1, 2, 3, 4, or 99).")

# Function to update the DataFrame and save to a CSV file
def update_and_save_dataframe(gaze_data, input_data, frame_count):
    gaze_data.loc[:,"hand_code"] = pd.Series(input_data)
    gaze_data.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv} at frame {frame_count}")

# Loop through each frame of the video, display to user, ask for code input, append input to DataFrame
while True:
    ret, frame = cap.read()

    # Function to handle end of file, or if any other runtime error is encountered
    if not ret:
        print("End of video or error reading frame.")
        break

    frame_count += 1

    # Resizes the frame window
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Frame", 810, 810)

    # Displays the frame to user
    cv2.imshow("Frame", frame)
    print(f"Frame {frame_count}")

    # Skips input on the first frame
    if frame_count > 1:
        # Get input from the user
        user_number = get_valid_number()
        input_data.append(user_number)

    # Updates and saves data to DataFrame every 100 frames
    if frame_count % 100 == 0:
        update_and_save_dataframe(gaze_data, input_data, frame_count)

    # Closes the frame window immediately if the user presses the ESC key
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        print("Exiting...")
        break

# Release video and close all windows
cap.release()
cv2.destroyAllWindows()

# Final update and save of the DataFrame
update_and_save_dataframe(gaze_data, input_data, frame_count)

# Calculates elapsed time for user to manually label the entire file, and displays elapsed time to user. Elapsed time is not saved anywhere automatically
et = time.time()
elapsed_min = (et - st) / 60
print("Time to complete file: "+str(elapsed_min))

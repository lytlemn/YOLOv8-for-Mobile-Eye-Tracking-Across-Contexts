# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:19:46 2024

@author: mnl5205
"""
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import pandas as pd
import datetime
import time
import glob
import random
import os

# Program for the automated labeling of video files, to be compared with manual labeling by hand-coding users

# Function to process our raw MET files dowloaded from Pupil cloud
def proc_met_files(subject):
    # Define folder path
    folder = os.path.join(
        '[raw data directory]',
        subject
    )

    # Find files
    gaze_file = next(glob.iglob(f"{folder}/**/gaze.csv"), None)
    timestamps_file = next(glob.iglob(f"{folder}/**/world_timestamps.csv"), None)
    events_file = next(glob.iglob(f"{folder}/**/events.csv"), None)

    # Read data
    events = pd.read_csv(events_file)
    gaze = pd.read_csv(gaze_file)
    timestamps = pd.read_csv(timestamps_file)

    # Turn events (markers identified by researcher and indicated in pupil cloud) into a dictionary
    events = events.rename(columns={"timestamp [ns]": "timestamp"}).set_index('name')['timestamp'].to_dict()

    # Prepare timestamps data
    timestamps['frame'] = timestamps.index
    timestamps = timestamps.drop(columns=['section id', 'recording id'])

    # Define task boundaries to filter gaze
    tasks = ['pprep', 'pspeech', 'cprep', 'cspeech']
    gaze_filt = pd.DataFrame()
    k = 0

    for task in tasks:
        start_key = f"{task}.start"
        end_key = f"{task}.end"

        if start_key in events and end_key in events:
            mask = (gaze['timestamp [ns]'] >= events[start_key]) & (gaze['timestamp [ns]'] <= events[end_key])
            task_gaze = gaze[mask].copy()
            task_gaze['task'] = task
            gaze_filt = pd.concat([gaze_filt, task_gaze])
            k +=1

    if k == 0:
        print("No task markers present, check data download")

    # Make sure gaze was corrected on Pupil cloud
    if "gaze.corrected" not in events:
        print("WARNING: No gaze correction indicator in events, check data tracker")

    # Merge gaze data with timestamps
    gaze_filt = pd.merge_asof(gaze_filt, timestamps, on="timestamp [ns]", direction='nearest')

    # Create a time column
    gaze_filt['time'] = (gaze_filt['timestamp [ns]'] - events['recording.begin']) / 1e9

    # Save the result
    out_fold = f"[Output data directory]/{subject}"
    os.makedirs(out_fold, exist_ok=True)
    output_file = os.path.join(out_fold, subject + '_gaze.csv')
    gaze_filt.to_csv(output_file, index=False)

    return gaze_filt

# Define a function which puts a circle for each gaze point on the current frame
def draw_gaze_points(frame, gazes, overlay_alpha=0.4):
    """Draw gaze points on a frame."""
    overlay = frame.copy()
    for x, y in gazes:
        coords = (int(x), int(y))
        cv2.circle(overlay, coords, color=(0, 128, 0), radius=30, thickness=cv2.FILLED)
        cv2.circle(overlay, coords, color=(0, 0, 255), radius=3, thickness=cv2.FILLED)
    return cv2.addWeighted(overlay, overlay_alpha, frame, 1 - overlay_alpha, 0)

# Function to determine if the face or person is within the "tv screen" bounding box
def is_inside_bbox(center, bbox):
    """Check if a point (center) is inside a bounding box."""
    x_min, y_min, x_max, y_max = map(int, bbox.xyxy.tolist()[0])
    return x_min < center[0] < x_max and y_min < center[1] < y_max

# Function to remove faces and people within our TV bounding boxes from our list of people/faces
def filter_entities(entities, tv_bboxes, track_ids_tvs, judges):
    """Filter entities (faces/people) not within TV bounding boxes."""
    keep_indices = []
    for entity in entities:
        entity_center = (
            (int(entity.xyxy.tolist()[0][0]) + int(entity.xyxy.tolist()[0][2])) // 2,
            (int(entity.xyxy.tolist()[0][1]) + int(entity.xyxy.tolist()[0][3])) // 2
        )
        inside_tv = False
        for bbox, track_id in zip(tv_bboxes, track_ids_tvs):
            if is_inside_bbox(entity_center, bbox):
                judges.append(track_id) # If there is a face is within the TV add this track id to the judges list
                inside_tv = True
                break
        keep_indices.append(0 if inside_tv else 1)
    return np.array(keep_indices), judges

def process_mask(mask, height, width):
    """Convert a mask to a boolean numpy array."""
    points = np.int32([mask.xy])
    img_black = np.zeros((height, width, 3), np.uint8)
    img_mask = cv2.fillPoly(img_black, points, (255, 255, 255))
    return np.all(img_mask == [255, 255, 255], axis=2)

# Function which determines if the gaze point touches the current segmentation mask
def check_gaze_intersection(frame_gazes, maskbool, height, width, face_mask=None):
    """Check gaze intersections with masks and optionally generate visuals."""
    gazein = np.full(len(frame_gazes), True)
    gazeinface = np.full(len(frame_gazes), True) if face_mask is not None else None

    for k, row in enumerate(frame_gazes):
        coords = (int(row[0]), int(row[1]))
        circlemask = np.zeros((height, width))
        cv2.circle(circlemask, coords, color=(1, 1, 1), radius=30, thickness=cv2.FILLED)
        circlemask[~maskbool] = 0
        gazein[k] = (circlemask.max() > 0)

        if face_mask is not None:
            circlemaskface = np.zeros((height, width))
            cv2.circle(circlemaskface, coords, color=(1, 1, 1), radius=30, thickness=cv2.FILLED)
            circlemaskface[~face_mask] = 0
            gazeinface[k] = (circlemaskface.max() > 0)

    return gazein, gazeinface

# Function which determines what to label the current gaze AOI based on gaze / mask intersections
def label_data(gazein, gazeinface, track_id, label_opts, faceloop):
    """Determine the label and track_id based on gaze intersections."""
    if faceloop:
        if gazeinface.all(): # If all of the gazes touch the face mask label as face
            label_opts[1] = track_id
        elif not gazeinface.all() and not (~gazeinface).all(): #If some in face mask and some outside label as uncodeable
            label_opts[99] = 0
        elif gazein.all(): # If all gaze points touch the body label as body
            label_opts[0] = track_id
        elif (~gazein).all(): # If all gaze points don't touch the body label as other
            label_opts[3] = 0
        else:
            label_opts[99] = 0 # Otherwise, label as uncodeable
    else:
        if gazein.all():
            label_opts[0] = track_id # If all gaze points touch the body label as body
        elif (~gazein).all():
            label_opts[3] = 0 # If all gaze points don't touch the body label as other
        else:
            label_opts[99] = 0 # Otherwise label as uncodeable

# Determine label based on priority
# For example if all gaze points touch the TV/Judge screen and body of the social partner
# We label that frame as body of social partner rather than TV/Judge screen
def finalize_label(label_opts):
    """Determine the final label and track based on label options."""
    for priority in [1, 0, 62, 3]:
        if priority in label_opts:
            return priority, label_opts[priority]
    return 99, 0

# Define the main function which calls above functions
def pcat_pipe(subject, gazes, vidgen, new):
    # Define paths to data and model files
    pt_path = './yolov8m-seg.pt'
    pt_path_face = './yolov8n-face.pt'

    folder = os.path.join(
        '[data folder]',
        subject
    )
    out_fold = os.path.join("[output folder]", subject)

    vid_path = next(glob.iglob(f"{folder}/**/*.mp4"), None)
    out_file2 = os.path.join(out_fold, subject + '.avi')
    out_hand = os.path.join(out_fold, subject + '_hand_out.avi')
    data_file = os.path.join(out_fold, subject + '_met_data.csv')

    # Save the temporarly file locally
    out_file = os.path.join(out_fold, subject + '_met_data_vidtemp.avi')

    # Load video
    cap = cv2.VideoCapture(vid_path)
    _, image = cap.read()
    height, width = image.shape[:2]

    if vidgen:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps<29.935: #Sometimes fps is low due to error at beginning of collection before task occured
            fps=29.94
        print(str(fps))
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
        outhand = cv2.VideoWriter(out_hand, fourcc, fps, (width, height))

    # Randomly select validation frames
    tasks = gazes['task'].unique()
    val_frames = []

    for task in tasks:
        task_frames = gazes.loc[gazes["task"] == task, "frame"].tolist()
        if len(set(task_frames))>500:
            start = random.randint(task_frames[0], task_frames[-1] - int(15 * fps))
            val_frames.extend(range(start, start + int(15 * fps) + 1))

    # Prepare output data
    outdata = gazes[['frame', 'task']].drop_duplicates()
    outdata['Code'] = 999
    outdata['Track_ID'] = -999
    outdata['Validation'] = 0

    all_frames = outdata['frame'].tolist()

    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Load YOLO models
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load primary model
    model = YOLO(pt_path).to(device)
    if torch.cuda.is_available():
        print("Using GPU")

    # Load face model
    model_face = YOLO(pt_path_face).to(device)

    # Initialize variables
    people_ids = []
    partner_ids = []
    judges = []
    non_tracker = -1

    # Log start time
    ct = datetime.datetime.now()
    st = time.time()
    print(f"Start time: {ct}")

    # Loop through every frame in the video
    for i in range(total_frames):
        if i % 3000 == 0:
            ct = datetime.datetime.now()
            print(f"time for frame number {i}", ct)
            outdata.to_csv(data_file, index=True)

        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        ret, frame = cap.read()

        if not ret:
            break

        # If the frame is within our task frames we process it
        if frame_num in all_frames:
            # Get gaze coordinates for the associated frame
            frame_gazes = gazes[gazes.frame == frame_num].iloc[:, 3:5].values

            # If this is one of our validation frames indicate it in data file and make a separate image to save to video with only gaze overlay
            if frame_num in val_frames:
                outdata.loc[outdata['frame'] == frame_num, ['Validation']] = 1
                hand_frame = frame.copy()

            # If there is no gaze for the current frame label as uncodeable
            if frame_gazes.size == 0:
                lab, track = 99, 0
            # Otherwise run the YOLO models on our frame
            else:
                output = model.track(frame, verbose=False, persist=True)
                bboxes = output[0].boxes
                nbboxes = len(bboxes)


                # If there is no detected things at all in the frame mark as looking at other and continue
                if nbboxes == 0:
                    lab, track = 3, 0

                elif output[0].masks is not None:
                    # Note 0=body, 1=face, 2=self, 3=other, 62=TV/judge, 99=uncodeable
                    # Refine arrays to just detected people and label as self vs body

                    masks = output[0].masks # Segmentation masks
                    pred_cls = bboxes.cls.cpu().numpy() # Predicted classes of the masks
                    pred_conf = bboxes.conf.cpu().numpy() # Predicted confidence of the object
                    # Filter to only include people and TVs of higher confidence
                    masks = masks[((pred_cls == 0) | (pred_cls == 62)) & (pred_conf > 0.35)]
                    bboxes = bboxes[((pred_cls == 0) | (pred_cls == 62)) & (pred_conf > 0.35)]
                    pred_cls= pred_cls[((pred_cls == 0) | (pred_cls == 62)) & (pred_conf > 0.35)]

                    # If there is a track id (tracks same object across frames) save it
                    # If not then create placeholders (usually when blurry image)
                    if bboxes.id is None:
                        # If the model didn't make track ids pick some random ones
                        track_ids = [i for i in range(non_tracker - len(masks),non_tracker)]
                        non_tracker = non_tracker - len(masks)
                        track_ids = np.array(track_ids)

                    else:
                        track_ids = bboxes.id.int().cpu().numpy()

                    if bboxes.id is None:
                        non_tracker -= len(masks)

                    # Label as other if no detected people or tvs
                    if len(pred_cls) == 0:
                        lab, track = 3, 0

                    elif len(pred_cls) != 0:

                         # Separate TVs and people
                        masks_peeps, bboxes_peeps, track_ids_peeps = masks[pred_cls == 0], bboxes[pred_cls == 0], track_ids[pred_cls == 0]
                        masks_tvs, bboxes_tvs, track_ids_tvs = masks[pred_cls == 62], bboxes[pred_cls == 62], track_ids[pred_cls == 62]
                        track_ids_tvs = track_ids_tvs.tolist()

                        # Run face detection if people are present
                        if len(masks_peeps) > 0:
                            output_face = model_face(frame, verbose=False)
                            bboxes_face = output_face[0].boxes
                            conf_face = bboxes_face.conf.cpu().numpy()

                            # Filter faces based on confidence
                            bboxes_face, conf_face = bboxes_face[conf_face > 0.35], conf_face[conf_face > 0.35]
                        else:
                            bboxes_face = []

                        # Filter out faces within TVs and label TVs as judges
                        if (len(bboxes_tvs) > 0) and (len(bboxes_face) > 0):
                            keep_faces, judges = filter_entities(bboxes_face, bboxes_tvs, track_ids_tvs, judges)
                            bboxes_face, conf_face = bboxes_face[keep_faces == 1], conf_face[keep_faces == 1]

                        # Filter out people within TVs
                        if (len(bboxes_tvs) > 0) and (len(bboxes_peeps) > 0):
                            keep_peeps, judges = filter_entities(bboxes_peeps, bboxes_tvs, track_ids_tvs, judges)
                            bboxes_peeps, masks_peeps, track_ids_peeps = (
                                bboxes_peeps[keep_peeps == 1],
                                masks_peeps[keep_peeps == 1],
                                track_ids_peeps[keep_peeps == 1]
                            )
                            track_ids_peeps = track_ids_peeps.tolist()

                        # Update people IDs
                        people_ids.extend(track_ids_peeps)
                        people_ids = list(set(people_ids))

                        # If there are faces present in the frame, create a mask for them based on upper part of body mask
                        if len(bboxes_face) > 0:

                            face = True
                            max_face = conf_face.argmax() # Get bounding box of highest confidence face (there can only be one person in room)
                            box_face = bboxes_face[max_face]

                            y_face = int(box_face.xyxy.tolist()[0][3])
                            x_face = int((int(box_face.xyxy.tolist()[0][0]) + int(box_face.xyxy.tolist()[0][2])) / 2)
                        else:
                            face = False
                            y_face = 0
                            x_face = 0

                        label_opts = {}

                        # Loop through people identified in the frame
                        for mask, bbox, track_id in zip(masks_peeps, bboxes_peeps, track_ids_peeps):
                            maskbool = process_mask(mask, height, width)
                            faceloop = face and maskbool[y_face-1, x_face] #If the face exists within this body

                            if faceloop:
                                face_mask = maskbool.copy()
                                face_mask[y_face:height, :] = False
                                maskbool[0:y_face, :] = False
                                partner_ids.append(track_id)

                                # Draw face_mask and obody mask on img
                                if vidgen:
                                    frame[face_mask] = frame[face_mask] * 0.5 + np.array([0, 87, 200], dtype=np.uint8) * 0.5
                                    frame[maskbool] = frame[maskbool] * 0.5 + np.array([205, 92, 92], dtype=np.uint8) * 0.5
                            else:
                                face_mask = None
                                if vidgen:
                                    frame[maskbool] = frame[maskbool] * 0.5 + np.array([205, 92, 92],
                                                                                       dtype=np.uint8) * 0.5

                            gazein, gazeinface = check_gaze_intersection(frame_gazes, maskbool, height, width, face_mask)
                            label_data(gazein, gazeinface, track_id, label_opts, faceloop)

                        # Loop through TVs
                        for mask, bbox, track_id in zip(masks_tvs, bboxes_tvs, track_ids_tvs):
                            try:
                                maskbool = process_mask(mask, height, width)
                                if vidgen:
                                    frame[maskbool] = frame[maskbool] * 0.5 + np.array([57, 255, 20], dtype=np.uint8) * 0.5
                                gazein, _ = check_gaze_intersection(frame_gazes, maskbool, height, width, None)
                                if gazein.all():
                                    label_opts[62] = track_id
                                elif (~gazein).all():
                                    label_opts[3] = 0
                                else:
                                    label_opts[99] = 0
                            except Exception as e: # This error occured for about 5 frames in our entire sample
                                print(e)
                                print("Error in calculating object overlay and gaze intersection for frame " + str(frame_num))
                                label_opts[99] = 0



                        lab, track = finalize_label(label_opts)
                else:
                    lab = 99
                    track = 0
            # Save output label and track id to the dataframe
            outdata.loc[outdata['frame'] == frame_num, ['Code']] = lab
            outdata.loc[outdata['frame'] == frame_num, ['Track_ID']] = track

            # Add gaze points and frame number to frame image and add to output video
            if vidgen:
                frame = draw_gaze_points(frame, frame_gazes, overlay_alpha=0.8)
                cv2.putText(frame, str(frame_num), (985, 30), 0, 1, [0, 0, 0], thickness=3)
                out.write(frame)
                if frame_num in val_frames:
                    hand_frame = draw_gaze_points(hand_frame, frame_gazes, overlay_alpha=0.8)
                    cv2.putText(hand_frame, str(frame_num), (985, 30), 0, 1, [0, 0, 0], thickness=3)
                    outhand.write(hand_frame)

    # Check to see which people ids were never identified as having a face and save those as self (code 2)
    partner_ids = list(set(partner_ids))
    self_ids = list(set(people_ids) - set(partner_ids))
    print("self ids:")
    print(self_ids)
    print("partner ids:")
    print(partner_ids)
    # Overwrite ids without a face as the self
    outdata.loc[(outdata.Track_ID.isin(self_ids)) & (outdata["Code"] == 0),'Code'] = 2
    # Overwrite ids with judge as judge
    outdata.loc[(outdata.Track_ID.isin(judges)) & (outdata["Code"] == 62),'Code'] = 4

    # Overwrite all tvs as judge for newer files in which there is only one monitor in the room
    if new:
        outdata.loc[(outdata["Code"] == 62),'Code'] = 4
    else:
        # Tidy up judge codes based on task (cannot see other tv when not giving speech), still may need some manual checking of 62 codes to label as 3 or 4
        if "C" in subject:
            outdata.loc[(outdata["task"].str.startswith('p')) & (outdata["Code"] == 62), 'Code'] = 4
        elif "P" in subject:
            outdata.loc[(outdata["task"].str.startswith('c')) & (outdata["Code"] == 62), 'Code'] = 4

    # Save output data
    outdata.to_csv(data_file, index=False)
    # Return time spent processing
    et = time.time()
    elapsed_min = (et - st) / 60
    print('Execution time: ', elapsed_min, ' minutes')
    # Add labels to our output video so we can get a visual of how well the model is doing
    if vidgen:
        out.release()
        outhand.release()
        # Add label to video
        # Load Video
        cap2 = cv2.VideoCapture(out_file)
        height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap2.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out2 = cv2.VideoWriter(out_file2, fourcc, fps, (width, height))
        total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(total_frames2):
            _, frame = cap2.read()  # Grab the frame'
            cv2.putText(frame, str(outdata.iloc[i,2]), (30, 30), 0, 1, [0, 0, 0], thickness=3)
            out2.write(frame)
        out2.release()
        

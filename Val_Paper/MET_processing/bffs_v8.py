from ultralytics import YOLO
import torch
import cv2
import numpy as np
import pandas as pd
import datetime
import time
import os
import random

# Program for the automatic labeling of video files, to be compared with the manual labeling by users

def bffs_pipe(subject, vidgen):
    # Define paths to data and model files
    pt_path = './yolov8m-seg.pt'
    pt_path_face = './yolov8n-face.pt'

    vid_path = '../data/BFFs/raw/' + subject + '.mp4'
    gaze_file = '../data/BFFs/raw/' + subject + '_gaze_positions.csv'

    out_file = '../data/BFFs/output/' + subject + '_met_data_vidtemp.avi'
    out_file2 = '../data/BFFs/output/' + subject + '.avi'
    data_file = '../data/BFFs/output/' + subject + '_met_data.csv'
    out_hand = '../data/BFFs/output/' + subject + '_hand_out.avi'

    # Load Video
    cap = cv2.VideoCapture(vid_path)
    _, image = cap.read()
    height, width = image.shape[:2]
    if vidgen:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps < 24.98: # Sometimes fps is low due to error at beginning of collection before task occured
            fps=24.98
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height)) # Create an output video for gaze and segmentation overlay
        outhand = cv2.VideoWriter(out_hand, fourcc, fps, (width, height)) # Create an output video with only gaze overlay for human labelling

    # Load eye gaze information for video
    # Important Note: This data was exported from Pupil Player in the window that only contained the task which contrasts to how data
    # were processed for PCAT. Therefore we don't have to filter for task related frames prior to processing
    gazes = pd.read_csv(gaze_file)  # Load gaze csv


    all_frames = [item for item in range(min(gazes['world_index'].unique()), max(gazes['world_index'].unique()) + 1)]

    # Select validation frames
    sections = np.array_split(all_frames,4)
    valframes = []
    for k in range(len(sections)):
        chunk = sections[k]
        # Randomly select the four 15 second segments
        if chunk.size != 0:
            if chunk.size <= 15*fps:
                valframes.extend(chunk)
            else:
                start = random.randint(chunk[0],chunk[-1] - int(15*fps))
                end = start + int(15*fps)
                chunk_frames = [num for num in range(start,end+1)]
                valframes.extend(chunk_frames)
    
    # Uncomment if only want to generate a 1 minute video for validation analyses:
    # all_frames = valframes  
    
    print(str(len(valframes)))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create DataFrame for output data
    outdata = pd.DataFrame(all_frames, columns=['Frame_Num'])
    outdata['Code'] = 999 # Load eye gaze information for video
    outdata['Face'] = 0
    outdata['Track_ID'] = -999

    # Load model
    torch.cuda.set_device(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(pt_path).to(device=device)

    if torch.cuda.is_available():
        print("using gpu")

    # Load face model
    model_face = YOLO(pt_path_face).to(device=device)

    # Create some lists to store people so we can identify which have faces (social partner)
    people_ids = []
    friend_ids = []
    new_ids = []
    non_tracker = -1

    # Print our start time
    ct = datetime.datetime.now()
    st = time.time()
    print("start time: ", ct)

    # Loop through all frames
    for i in range(total_frames):

        # Every 1000 frames print the current time and frame number and write temp data file
        if i % 1000 ==0:
            ct = datetime.datetime.now()
            print("time for frame number", i, ct)
            outdata.to_csv(data_file, index=True)

        frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        _, frame = cap.read()  # Grab the frame'

        # If it is one of the frames we want to process
        if frame_num in all_frames:
            # Get the gaze coordinates for that current frame from our gaze data
            frame_gazes = gazes[gazes.world_index == frame_num].iloc[:, 3:5].values

            # If there are no gaze coordinates label as uncodeable
            if len(frame_gazes) == 0:
                lab = 99
            else:
            # Increase contrast of image to help with washing out due to too high brightness setting at time of data collection
                lab= cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l_channel, a, b = cv2.split(lab)

                # Applying CLAHE to L-channel
                # Feel free to try different values for the limit and grid size:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                cl = clahe.apply(l_channel)

                # Merge the CLAHE enhanced L-channel with the a and b channel
                limg = cv2.merge((cl,a,b))

                # Converting image from LAB Color model to BGR color spcae
                frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

                image = frame.copy()
                # Run our frame through the YOLOv8 segmentation model
                output = model.track(image, verbose=False, persist=True)

                bboxes = output[0].boxes
                nbboxes = len(bboxes)

                if vidgen:
                    pnimg = image.copy()
                    imageeyes = image.copy()

                # If there is no detected things at all in the frame mark as looking at other and continue
                if nbboxes == 0:
                    lab = 3
                    track = 0
                    # Plot the eye gaze on video output
                    if vidgen:
                        for k, row in enumerate(frame_gazes):
                            coords = (int(row[0] * width), int(((1 - row[1]) * height)))
                            overlay = pnimg.copy()
                            cv2.circle(overlay, coords, color=(0, 128, 0), radius=20, thickness=cv2.FILLED)
                            cv2.circle(overlay, coords, color=(0, 0, 255), radius=3, thickness=cv2.FILLED)
                            pnimg = cv2.addWeighted(overlay, 0.4, pnimg, 0.6, 0)
                            overlay2 = imageeyes.copy()
                            cv2.circle(overlay2, coords, color=(0, 128, 0), radius=20, thickness=cv2.FILLED)
                            cv2.circle(overlay2, coords, color=(0, 0, 255), radius=3, thickness=cv2.FILLED)
                            imageeyes = cv2.addWeighted(overlay2, 0.4, imageeyes, 0.6, 0)
                # If there were detected objects
                elif output[0].masks is not None:
                    # Codes Note: 0=body, 1=face, 2=self, 3=other, 99=uncodeable

                    masks = output[0].masks # Segmentation masks
                    pred_cls = bboxes.cls.cpu().numpy() # Predicted classes of the masks
                    pred_conf = bboxes.conf.cpu().numpy() # Predicted confidence of the object

                    # Filter to only include people of higher confidence
                    masks = masks[(pred_cls == 0) & (pred_conf > 0.35)]
                    bboxes = bboxes[(pred_cls == 0) & (pred_conf > 0.35)]
                    pred_cls = pred_cls[(pred_cls == 0) & (pred_conf > 0.35)]
                    pred_cls = pred_cls.tolist()

                    # If there is a track id (tracks same object across frames) save it
                    # If not then create placeholders (usually when blurry image)
                    if output[0].boxes.id is None:
                        # If the model didn't make track ids pick some random ones
                        track_ids = [i for i in range(non_tracker - len(masks),non_tracker+1)]
                        non_tracker = non_tracker - len(masks) - 1
                    else:
                        track_ids = bboxes.id.int().cpu().numpy()
                        track_ids = track_ids.tolist()


                    # Label as other if no detected people
                    if len(pred_cls) == 0:
                        lab = 3
                        track = 0
                        if vidgen:
                            for k, row in enumerate(frame_gazes):
                                coords = (int(row[0] * width), int(((1 - row[1]) * height)))
                                overlay = pnimg.copy()
                                cv2.circle(overlay, coords, color=(0, 128, 0), radius=20, thickness=cv2.FILLED)
                                cv2.circle(overlay, coords, color=(0, 0, 255), radius=3, thickness=cv2.FILLED)
                                pnimg = cv2.addWeighted(overlay, 0.4, pnimg, 0.6, 0)
                                overlay2 = imageeyes.copy()
                                cv2.circle(overlay2, coords, color=(0, 128, 0), radius=20, thickness=cv2.FILLED)
                                cv2.circle(overlay2, coords, color=(0, 0, 255), radius=3, thickness=cv2.FILLED)
                                imageeyes = cv2.addWeighted(overlay2, 0.4, imageeyes, 0.6, 0)

                    elif len(pred_cls) != 0:

                        # Add track_ids to all people and then remove duplicates
                        people_ids.extend(track_ids)
                        people_ids = list(set(people_ids))

                        # Run face detection
                        output_face = model_face(image, verbose=False)
                        bboxes_face = output_face[0].boxes
                        conf_face = bboxes_face.conf.cpu().numpy()

                        bboxes_face = bboxes_face[(conf_face > 0.5)]
                        conf_face = conf_face[(conf_face > 0.5)]

                        if len(bboxes_face) > 0:
                            face = True
                            outdata.loc[outdata['Frame_Num'] == frame_num, ['Face']] = 1
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
                        for mask, bbox, cl, track_id  in zip(masks, bboxes, pred_cls, track_ids):

                            gazein = np.full(len(frame_gazes), True)
                            # Convert our mask into a numpy array
                            points = np.int32([mask.xy])
                            img_black = np.zeros((height, width, 3), np.uint8)
                            img_mask = cv2.fillPoly(img_black, points, (255, 255, 255))
                            # Create a boolean array which identifies which pixels the detected person object occupies
                            maskbool = np.alltrue(img_mask == [255, 255, 255], axis=2)

                            # If the face exists and the bottom middle point of the face is within the body
                            if (cl == 0) & face & (maskbool[y_face-1,x_face]):
                                faceloop = True
                                gazeinface = np.full(len(frame_gazes), True)
                                # Split mask in half along face
                                face_mask = maskbool.copy()
                                # Below face ypos false
                                face_mask[y_face:height, :] = False
                                # Above face ypos false
                                maskbool[0:y_face, :] = False

                                # Assign this track_id as friend
                                friend_ids.append(track_id)
                                friend_ids = list(set(friend_ids))

                                # Draw face_mask and body mask on img
                                if vidgen:
                                    pnimg[face_mask] = pnimg[face_mask] * 0.5 + np.array([0, 87, 200], dtype=np.uint8) * 0.5
                                    pnimg[maskbool] = pnimg[maskbool] * 0.5 + np.array([205, 92, 92], dtype=np.uint8) * 0.5

                            else:
                                faceloop = False
                                if vidgen:
                                    pnimg[maskbool] = pnimg[maskbool] * 0.5 + np.array([205, 92, 92],
                                                                                       dtype=np.uint8) * 0.5
                            # Plot each of our frame gazes and determine if they touch any of the identified people
                            for k, row in enumerate(frame_gazes):
                                coords = (int(row[0] * width), int(((1 - row[1]) * height))) # Gaze coordinates
                                circlemask = np.zeros((height, width))
                                cv2.circle(circlemask, coords, color=(1, 1, 1), radius=20, thickness=cv2.FILLED)
                                circlemask[~maskbool] = circlemask[~maskbool] * 0
                                gazein[k] = (1 in circlemask)  # Indicates gaze point intersects with identified person
                                if faceloop:
                                    circlemaskface = np.zeros((height, width))
                                    cv2.circle(circlemaskface, coords, color=(1, 1, 1), radius=20, thickness=cv2.FILLED)
                                    circlemaskface[~face_mask] = circlemaskface[~face_mask] * 0
                                    gazeinface[k] = (1 in circlemaskface) # Indicates gaze point intersects with identified face
                                    # Add dot to image
                                if vidgen:
                                    overlay = pnimg.copy()
                                    cv2.circle(overlay, coords, color=(0, 128, 0), radius=20, thickness=cv2.FILLED)
                                    cv2.circle(overlay, coords, color=(0, 0, 255), radius=3, thickness=cv2.FILLED)
                                    pnimg = cv2.addWeighted(overlay, 0.4, pnimg, 0.6, 0)
                                    overlay2 = imageeyes.copy()
                                    cv2.circle(overlay2, coords, color=(0, 128, 0), radius=20, thickness=cv2.FILLED)
                                    cv2.circle(overlay2, coords, color=(0, 0, 255), radius=3, thickness=cv2.FILLED)
                                    imageeyes = cv2.addWeighted(overlay2, 0.4, imageeyes, 0.6, 0)
                            # Check if all gaze positions intersect with the current body mask/face or not
                            if faceloop:
                                if all(gazeinface):  # Start with face as overrides body
                                    label_opts.update({1:track_id})
                                elif (not all(gazeinface)) & (not all(~gazeinface)):  # If some in face and some not label as uncodeable
                                    label_opts.update({99:0})
                                elif all(gazein):
                                    label_opts.update({cl:track_id})  # Will label as body if all in there
                                elif all(~gazein):  # If all not in face or body segmentation will label as other
                                    label_opts.update({3:0})
                                else:  # If some not in the current mask will label as uncodeable
                                    label_opts.update({99:0})
                            elif not faceloop:
                                if all(gazein):
                                    label_opts.update({cl:track_id})  # Will label as self or body if all in there
                                elif all(~gazein):  # If all not in mask will label as other
                                    label_opts.update({3:0})
                                else:  # If some not in mask will label as uncodeable
                                    label_opts.update({99:0})
                        # After looping through all all masks determine if all gaze points point to the same label
                        if 1 in list(label_opts.keys()):
                            lab = 1
                            track = label_opts.get(1)
                        elif 0 in list(label_opts.keys()):
                            lab = 0
                            track = label_opts.get(0)
                        elif 3 in list(label_opts.keys()):
                            lab = 3
                            track = 0
                        else:
                            lab = 99
                            track = 0
                else:
                    lab = 99
                    track = 0

            # Add our labels and track ids to our output dataframe
            outdata.loc[outdata['Frame_Num'] == frame_num, ['Code']] = lab
            outdata.loc[outdata['Frame_Num'] == frame_num, ['Track_ID']] = track

            # Write our images pnimg (with segmentation and gaze overlay) and imgeyes (only gaze overlay) to our video ouput
            if vidgen:
                out.write(pnimg)
                cv2.putText(imageeyes, str(frame_num), (985, 30), 0, 1, [0, 0, 0], thickness=3)
                outhand.write(imageeyes)

    # Check to see which people ids were never identified as having a face and save those as self (code 2)
    self_ids = list(set(people_ids) - set(friend_ids))
    print("self ids:")
    print(self_ids)
    print("friend ids:")
    print(friend_ids)
    outdata.loc[outdata.Track_ID.isin(self_ids),'Code'] = 2
    outdata.to_csv(data_file, index=False)
    et = time.time()
    elapsed_min = (et - st) / 60

    # Displays the program's elapsed time for completing file labeling. Can be compared with manual user elapsed time for hand labeling
    print('Execution time: ', elapsed_min, ' minutes')
    
    # Add labels to our output video so we can get a visual of how well the model is doing
    if vidgen:
        out.release()
        outhand.release()

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
            cv2.putText(frame, str(outdata.loc[i,'Code']), (30, 30), 0, 1, [0, 0, 0], thickness=3)
            out2.write(frame)
        out2.release()

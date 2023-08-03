import os
import cv2
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import json
class Video:
    def __init__(self, file_dir):
        self.Video_List = glob.glob(os.path.join(file_dir, "*.avi"))  # absolute paths
        self.Video_names = [os.path.splitext(os.path.basename(f))[0] for f in self.Video_List]  # names without extensions
        self.videos = [cv2.VideoCapture(f) for f in self.Video_List]

    def info(self):
        info = {
            "Video_List": self.Video_List,
            "Video_names": self.Video_names,
            "metaInfo": {
                "Video Number": len(self.videos),
                "Total Time": sum(video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS) for video in self.videos),
                "Resolution": [(int(video.get(3)), int(video.get(4))) for video in self.videos]
            }
        }
        return info

    from tqdm import tqdm

    def to_Frame(self, target_dir):
        # Calculate total frames in all videos to set up the progress bar
        total_frames = sum(int(video.get(cv2.CAP_PROP_FRAME_COUNT)) for video in self.videos)
        pbar = tqdm(total=total_frames)

        for video_path, video in zip(self.Video_List, self.videos):
            # Create target directory for each video with the same base name as the video file, excluding the extension
            video_target_dir = os.path.join(target_dir, os.path.splitext(os.path.basename(video_path))[0])
            if not os.path.exists(video_target_dir):
                os.makedirs(video_target_dir)

            # Convert each frame of the video into an image file
            success, image = video.read()
            while success:
                cv2.imwrite(os.path.join(video_target_dir, "{}.jpg".format(int(video.get(cv2.CAP_PROP_POS_MSEC)))), image)
                success, image = video.read()
                pbar.update(1)  # update progress bar

        pbar.close()

 
    def Calculate_Minimal_Box(self, Frame_address):
        # Get a list of all the subdirectories
        subdirectories = [f.path for f in os.scandir(Frame_address) if f.is_dir()]

        # Iterate over each subdirectory
        for subdir in tqdm(subdirectories):
            # Construct the path to the '0.jpg' file in this subdirectory
            image_path = os.path.join(subdir, '0.jpg')

            # Load the image using OpenCV
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # assuming the image is grayscale

            # Find the coordinates of non-zero pixels
            non_zero_pixels = cv2.findNonZero(img)

            # Calculate the bounding box
            x, y, w, h = cv2.boundingRect(non_zero_pixels)

            # Store the bounding box in a dictionary
            bounding_box = {'x': x, 'y': y, 'w': w, 'h': h}

            # Write the bounding box data to a JSON file in the same subdirectory
            with open(os.path.join(subdir, 'Bounding_Box.json'), 'w') as outfile:
                json.dump(bounding_box, outfile)

        print("Bounding boxes calculated and saved to 'Bounding_Box.json' in each subdirectory.")

class AutoEngagement:
    def __init__(self, dir_path=None):
        if dir_path is not None:
            self.Load_Video(dir_path)

    def Load_Video(self, file_dir):
        return Video(file_dir)


    def Load_Image(self, file_dir):
        # Gather all the generated image files across all directories
        image_paths = []
        for dirpath, dirnames, filenames in os.walk(file_dir):
            for filename in [f for f in filenames if f.endswith(".jpg")]:
                image_paths.append(os.path.join(dirpath, filename))

        # Create dictionary of video number, frame, and address
        frames = {}
        for image_path in image_paths:
            dir_name, image_name = os.path.split(image_path)
            video_num = os.path.basename(dir_name)
            frame_num = os.path.splitext(image_name)[0]
            if video_num not in frames:
                frames[video_num] = []
            frames[video_num].append({
                "Frame": frame_num,
                "Address": image_path
            })

        # Transform the dictionary into a MultiIndex DataFrame
        frames_list = [(video_num, frame_info['Frame'], frame_info['Address']) for video_num in frames for frame_info in frames[video_num]]
        df = pd.DataFrame(frames_list, columns=['Video_Num', 'Frame', 'Address'])


        return df
    

    def Load_Label(self, Label_address):
        # Read the header
        import csv
        with open(Label_address, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

        # Find the maximum number of columns
        with open(Label_address, 'r') as f:
            max_columns = max(len(list(row)) for row in csv.reader(f))

        # If the header is shorter than the maximum number of columns, extend it
        if len(header) < max_columns:
            header.extend([f'col{i}' for i in range(len(header), max_columns)])

        # Load the CSV into a DataFrame using the header
        df = pd.read_csv(Label_address, names=header, skiprows=1)

        # Create a new DataFrame to hold the restructured data
        new_df = pd.DataFrame(columns=['Video_Num', 'Start', 'End', 'Label'])

        # Iterate over each row in the DataFrame
        for i, row in tqdm(df.iterrows(), total=df.shape[0]):
            # Get the basic information
            video_num_with_ext = row['video ID']
            video_num, _ = os.path.splitext(video_num_with_ext)

            # Find the index where the timestamps start
            stamps_start_index = df.columns.get_loc("time")

            # Find the index where the labels start
            label_start_index = next((index for index, value in enumerate(row.values) if index > stamps_start_index and value < row.values[index-1]), None)

            # Determine length of timestamps and labels section
            length = label_start_index - stamps_start_index

            if row[label_start_index+length-1] == np.nan:
                raise ValueError(f"Row length{length} doesn't match expected total length{len(row)} with stamps {stamps_start_index + 1}, label {label_start_index} for video {video_num}")
                #return row    
            # Iterate over the timestamps and corresponding labels
            for j in range(length):
                # Get the start time and label
                start = row[j + stamps_start_index]

                # If it's the last timestamp, the end time should be 100000
                if j != length-1:
                    end = row[j + stamps_start_index + 1]
                else:
                    end = 100000

                # Get the label
                label = row[j + label_start_index]

                # Append to the new DataFrame
                new_row = pd.DataFrame({'Video_Num': [video_num], 'Start': [start], 'End': [end], 'Label': [label]})
                new_df = pd.concat([new_df, new_row], ignore_index=True)

        return new_df


    def Merge_Label(self, frames, labels):
        # Convert the columns to integers
        frames['Frame'] = frames['Frame'].astype(int)
        labels['Start'] = labels['Start'].astype(int)
        labels['End'] = labels['End'].astype(int)

        # Create a new DataFrame to hold the merged data
        merged_df = pd.DataFrame(columns=['Video_Num', 'Frame', 'Address', 'Label'])

        # Iterate over each row in the frames DataFrame
        for _, frame_row in tqdm(frames.iterrows(), total=frames.shape[0]):
            # Get the frame's video number and frame number
            frame_video_num = frame_row['Video_Num']
            frame_num = frame_row['Frame']

            # Find matching labels
            matching_labels = labels[(labels['Video_Num'] == frame_video_num) &
                                     (labels['Start'] <= frame_num) & 
                                     (labels['End'] > frame_num)]
            if not matching_labels.empty:
                # If there are matching labels, take the first one
                label = matching_labels.iloc[0]['Label']

                # Append the information to the new DataFrame
                new_row = pd.DataFrame({'Video_Num': [frame_video_num], 'Frame': [frame_num], 'Address': [frame_row['Address']], 'Label': [label]})
                merged_df = pd.concat([merged_df, new_row], ignore_index=True)

        return merged_df
########################################################
# my_dataset.py
########################################################

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

########################################################
# EXAMPLE: Advisor's 6 videos approach
# You can adapt or skip if you have your own listing.
########################################################
VIDEOS = [
    {
        "name": "10 Things Jack Harlow Can't Live Without  GQ",
        "class": "Non-PTSD",  # or 0/1 for label
        "video_dir": "/Users/caleb/Saving_Directory/Visual/Non-PTSD/10 Things Jack Harlow Can't Live Without  GQ",
        "audio_dir": "/Users/caleb/Audio_Saving_Directory/Non-PTSD/10 Things Jack Harlow Can't Live Without  GQ",
        "num_video_segments": 47,
        "fps": 23.976,
        "segment_frames": 300
    },
    {
        "name": "10 Things Joji Can't Live Without  GQ",
        "class": "Non-PTSD",
        "video_dir": "/Users/caleb/Saving_Directory/Visual/Non-PTSD/10 Things Joji Can't Live Without  GQ",
        "audio_dir": "/Users/caleb/Audio_Saving_Directory/Non-PTSD/10 Things Joji Can't Live Without  GQ",
        "num_video_segments": 45,
        "fps": 23.976,
        "segment_frames": 300
    },
    {
        "name": "10 Things Odell Beckham Jr. Can't Live Without  GQ Sports",
        "class": "Non-PTSD",
        "video_dir": "/Users/caleb/Saving_Directory/Visual/Non-PTSD/10 Things Odell Beckham Jr. Can't Live Without  GQ Sports",
        "audio_dir": "/Users/caleb/Audio_Saving_Directory/Non-PTSD/10 Things Odell Beckham Jr. Can't Live Without  GQ Sports",
        "num_video_segments": 52,
        "fps": 23.976,
        "segment_frames": 300
    },
    {
        "name": "vidm001",
        "class": "PTSD",
        "video_dir": "/Users/caleb/Saving_Directory/Visual/PTSD/vidm001",
        "audio_dir": "/Users/caleb/Audio_Saving_Directory/PTSD/vidm001",
        "num_video_segments": 27,
        "fps": 23.976,
        "segment_frames": 300
    },
    {
        "name": "vidm049",
        "class": "PTSD",
        "video_dir": "/Users/caleb/Saving_Directory/Visual/PTSD/vidm049",
        "audio_dir": "/Users/caleb/Audio_Saving_Directory/PTSD/vidm049",
        "num_video_segments": 27,
        "fps": 23.976,
        "segment_frames": 300
    },
    {
        "name": "vidm080",
        "class": "PTSD",
        "video_dir": "/Users/caleb/Saving_Directory/Visual/PTSD/vidm080",
        "audio_dir": "/Users/caleb/Audio_Saving_Directory/PTSD/vidm080",
        "num_video_segments": 29,
        "fps": 23.976,
        "segment_frames": 300
    }
]


########################################################
# DataArranger: enumerates your new data structure
########################################################

from base.dataset import GenericDataArranger

class DataArranger(GenericDataArranger):
    """
    An updated DataArranger that enumerates each video
    and all its segment_### subfolders. We skip the original
    "partition_range_fn()" approach in favor of a simpler listing.
    Or we keep it if you prefer fold-based partition logic.
    """
    def __init__(self, debug=0):
        """
        If you have some dataset_info, pass it. Otherwise we skip.
        `debug` might reduce the number of segments for quick testing.
        """
        self.debug = debug
        # We'll store a "data_list" with info about each segment
        self.data_list = []
        self.partition_dict = {"train": [], "validate": [], "test": []}  # or whichever splits

        self.prepare_data_list()

    def prepare_data_list(self):
        """
        We'll do a simple approach: put all segments in 'train' for example,
        or do a ratio-based split. If you want more sophisticated splitting, adapt.
        """
        for vid_info in VIDEOS:
            video_dir = vid_info["video_dir"]
            audio_dir = vid_info["audio_dir"]
            label_str = vid_info["class"]   # "PTSD" or "Non-PTSD"
            # Let's assign label=1 for PTSD, label=0 for Non-PTSD
            label_val = 1 if label_str == "PTSD" else 0

            fps = vid_info["fps"]
            frames_per_seg = vid_info["segment_frames"]
            seg_duration_sec = frames_per_seg / fps

            num_segments = vid_info["num_video_segments"]
            if self.debug and num_segments>5:
                num_segments = 5  # shorter if debug=1

            for i in range(num_segments):
                seg_folder = f"segment_{i:03d}"
                segment_path = os.path.join(video_dir, seg_folder)
                faces_dir = os.path.join(segment_path, "faces")
                # If we assume there's an "audio.npy" or something
                # e.g. `segment_00i.npy`, adapt as needed
                audio_file = os.path.join(audio_dir, f"segment_{i+1}.npy")  # or i if you prefer

                start_sec = i * seg_duration_sec
                end_sec   = start_sec + seg_duration_sec

                item = {
                    "segment_path": segment_path,   # for video
                    "faces_dir": faces_dir,
                    "audio_file": audio_file,        # if you have direct 12.5s audio chunk
                    "label": label_val,
                    "start_sec": start_sec,
                    "end_sec":   end_sec
                }
                # For simplicity, put everything in "train"
                # or do a random split. Here we do "train" for demonstration
                self.partition_dict["train"].append(item)

        # you could do more logic for "validate", "test"

    def get_partitioned_data_list(self, partition="train"):
        """
        Return the data_list for the given partition.
        In the old code, it was 'partition_range'. Now we store in self.partition_dict
        """
        return self.partition_dict.get(partition, [])


########################################################
# The actual dataset that loads from these segments
########################################################

from base.dataset import GenericDataset

class Dataset(GenericDataset):
    """
    A custom dataset that reads from your new data storing format:
     - segment_path/faces/*.jpg for frames
     - audio_file for audio chunk
     - label for PTSD vs. Non-PTSD
    """
    def __init__(self, data_list, transform=None):
        super().__init__(
            data_list=data_list,
            modality=['video','audio'],  # if the base code requires
            multiplier={},
            feature_dimension={},
            window_length=0,
            mode='train'
        )
        self.data_list = data_list
        self.transform = transform

    def __getitem__(self, idx):
        item = self.data_list[idx]
        faces_dir = item["faces_dir"]
        audio_file= item["audio_file"]
        label     = item["label"]
        start_sec = item["start_sec"]
        end_sec   = item["end_sec"]

        # 1) Load frames
        frames_list = []
        if os.path.isdir(faces_dir):
            frame_files = sorted(f for f in os.listdir(faces_dir) if f.endswith('.jpg'))
            for ff in frame_files:
                fpath = os.path.join(faces_dir, ff)
                frame_bgr = cv2.imread(fpath)
                # optionally convert to RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames_list.append(frame_rgb)  # shape (H,W,3)
        else:
            # no faces folder => empty?
            frames_list = []

        # 2) Load audio chunk
        audio_array = None
        if os.path.isfile(audio_file):
            arr = np.load(audio_file)
            # if arr is 2D => (N,64) => expand to (N,1,64) if needed
            if arr.ndim == 2:
                arr = arr[:, None, :]
            audio_array = arr
        else:
            # maybe produce empty or zeros
            audio_array = np.zeros((1,64,64), dtype=np.float32)

        # 3) Possibly transform frames or audio into tensors
        # We'll keep them as is for demonstration
        # If you want PyTorch Tensors:
        # frames_tensor = [torch.from_numpy(f).permute(2,0,1) for f in frames_list]
        # audio_tensor = torch.from_numpy(audio_array)

        out_dict = {
            "video": frames_list,
            "audio": audio_array,
            "label": label
        }
        return out_dict

    def __len__(self):
        return len(self.data_list)


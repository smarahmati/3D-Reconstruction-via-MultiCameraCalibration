"""
DeepLabCut Project Creation for DLC 2.3.11
------------------------------------------

This script demonstrates:
1) Creating a DLC project from either:
   - A real video (and extracting frames).
   - A dummy video (made from images), if you have no real video.
2) Skipping extraction if you're only using pre-extracted images (i.e., the dummy scenario).
3) Removing default DLC placeholders (bodypart1, bodypart2, bodypart3, objectA).
4) Adding user-defined body parts to the config.
5) Demonstrating uniform sampling for 'extract_frames' by updating the config.yaml
   with 'start', 'stop', and 'numframes2pick'.

Author: [Your Name]
"""

import os
import shutil
import cv2
import deeplabcut
from deeplabcut.utils import auxiliaryfunctions

# -----------------------------------------------------------------------------
# User Settings
# -----------------------------------------------------------------------------

project_name = "Pendulum-Cam3"
experimenter_name = "Rahmati"
project_path = r"C:\Users\smara\Desktop"

# Use a real video:
video_path = r"C:\Users\smara\Desktop\Cam3_2025-04-21 22-29-36.mkv"
# Or, if you only have images:
images_folder_path = None  # e.g., r"C:\Users\smara\Videos\Cam2"

dummy_video_filename = "dummy.mp4"
dummy_video_fps = 5

# Body parts you want to track
body_parts = [
    "P01", "P02", "P03"
# "P00", "P01", "P02", "P03", "P04",
# "P10", "P11", "P12", "P13", "P14"
]

# For uniform sampling:
# We'll pick 20 frames from 0% to 100% of the video, evenly spaced.
uniform_numframes = 50    # e.g., 20 frames
uniform_start = 0.0       # fraction of video start
uniform_stop = 1.0        # fraction of video end

# For k-means (as a comment/example):
# cluster_step = 5    # skip every 5 frames before clustering
# (We'd still rely on config['numframes2pick'] for total frames picked.)

# -----------------------------------------------------------------------------
# End User Settings
# -----------------------------------------------------------------------------


def create_dummy_video_from_images(images_dir, output_video_path, fps=5):
    """Creates a dummy .mp4 from sorted images in `images_dir`."""
    if not os.path.isdir(images_dir):
        raise ValueError(f"Images folder not found: {images_dir}")

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])
    if not image_files:
        raise ValueError(f"No valid image files found in folder: {images_dir}")

    first_frame = cv2.imread(os.path.join(images_dir, image_files[0]))
    if first_frame is None:
        raise ValueError("Cannot read the first image.")

    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        frame = cv2.imread(img_path)
        if frame is not None:
            writer.write(frame)
        else:
            print(f"Warning: cannot read file {img_path}")

    writer.release()
    print(f"Dummy video created at: {output_video_path}")
    return output_video_path


def update_bodyparts_in_config(config_path, user_bodyparts):
    """Remove default placeholders and add user-defined body parts."""
    if not os.path.isfile(config_path):
        print(f"No config.yaml found at: {config_path}")
        return

    cfg = auxiliaryfunctions.read_config(config_path)
    if "bodyparts" not in cfg or cfg["bodyparts"] is None:
        cfg["bodyparts"] = []

    # Remove DLC placeholders
    defaults_to_remove = ["bodypart1", "bodypart2", "bodypart3", "objectA"]
    for default_bp in defaults_to_remove:
        while default_bp in cfg["bodyparts"]:
            cfg["bodyparts"].remove(default_bp)

    # Add user-defined parts
    for bp in user_bodyparts:
        if bp not in cfg["bodyparts"]:
            cfg["bodyparts"].append(bp)

    auxiliaryfunctions.write_config(config_path, cfg)
    print("Updated config.yaml. Bodyparts:", cfg["bodyparts"])


def set_uniform_sampling_params(config_path, start=0.0, stop=1.0, numframes=20):
    """
    For uniform sampling in DLC 2.3.11, you must set:
    config['start'], config['stop'], config['numframes2pick'] in the config.yaml.
    """
    if not os.path.isfile(config_path):
        print(f"No config.yaml found at: {config_path}")
        return

    cfg = auxiliaryfunctions.read_config(config_path)
    cfg["start"] = start
    cfg["stop"] = stop
    cfg["numframes2pick"] = numframes  # how many frames to pick uniformly
    auxiliaryfunctions.write_config(config_path, cfg)
    print(f"Set uniform sampling in config: start={start}, stop={stop}, numframes2pick={numframes}")


def create_dlc_project(
    project_name,
    experimenter_name,
    project_path,
    video_path=None,
    images_folder_path=None,
    body_parts=None,
    uniform_start=0.0,
    uniform_stop=1.0,
    uniform_numframes=20,
    dummy_video_filename="dummy.mp4",
    dummy_video_fps=5
):
    """Creates a DLC project from either a real video or a dummy video from images."""
    if not project_name:
        raise ValueError("Project name is required.")
    if not experimenter_name:
        raise ValueError("Experimenter name is required.")
    if not os.path.isdir(project_path):
        raise ValueError("Invalid project_path. Folder not found.")

    # Decide if we use a real video
    use_real_video = (video_path and os.path.isfile(video_path))
    if use_real_video:
        final_video_path = video_path
        print(f"Using real video: {final_video_path}")
    else:
        if images_folder_path and os.path.isdir(images_folder_path):
            print("No real video. Creating dummy video from images...")
            dummy_path = os.path.join(project_path, dummy_video_filename)
            final_video_path = create_dummy_video_from_images(images_folder_path, dummy_path, fps=dummy_video_fps)
        else:
            raise ValueError("No valid video_path or images_folder_path provided.")

    # Create the DLC project
    print("Creating new DLC project with video:", final_video_path)
    config_path = deeplabcut.create_new_project(
        project_name,
        experimenter_name,
        [final_video_path],
        working_directory=project_path,
        copy_videos=False
    )
    if not config_path or config_path == 'nothingcreated':
        print("DLC did not create the project (check logs).")
        return None

    print(f"Project created. Config path: {config_path}")

    if use_real_video:
        # (1) We update the config for uniform sampling
        set_uniform_sampling_params(config_path, start=uniform_start, stop=uniform_stop, numframes=uniform_numframes)

        # (2) Then we call extract_frames using 'uniform'
        print("Extracting frames from the real video with uniform sampling.")
        deeplabcut.extract_frames(
            config_path,
            mode='automatic',
            algo='uniform',
            crop=False,
            userfeedback=False
            # Note: 'start', 'stop', 'numframes2pick' are read from config,
            # so we do not pass them here.
        )
        print("Uniform frame extraction complete.")

        # If you wanted k-means, you'd do something like:
        #   # set numframes2pick in config, and optionally cluster_step, cluster_color, etc.
        #   # Then call:
        #   deeplabcut.extract_frames(config_path, mode='automatic', algo='kmeans', cluster_step=5, ...)
    else:
        # Images-only: skip extraction
        print("Skipping frame extraction (images already exist).")
        import_dir = os.path.join(os.path.dirname(config_path), "labeled-data", "imported_images")
        os.makedirs(import_dir, exist_ok=True)
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        for img_name in sorted(os.listdir(images_folder_path)):
            if os.path.splitext(img_name)[1].lower() in valid_exts:
                shutil.copy(os.path.join(images_folder_path, img_name), os.path.join(import_dir, img_name))
        print(f"Images copied to {import_dir}.")

        # Remove the dummy-labeled-data folder
        dummy_base = os.path.splitext(os.path.basename(final_video_path))[0]
        dummy_label_folder = os.path.join(os.path.dirname(config_path), "labeled-data", dummy_base)
        if os.path.isdir(dummy_label_folder):
            shutil.rmtree(dummy_label_folder)
            print(f"Removed dummy-labeled-data folder: {dummy_label_folder}")

    # Remove default placeholders and add user-defined body parts
    if body_parts:
        print("Updating config.yaml with user-defined body parts...")
        update_bodyparts_in_config(config_path, body_parts)

    print("DLC project setup complete.")
    return config_path


if __name__ == "__main__":
    cfg = create_dlc_project(
        project_name=project_name,
        experimenter_name=experimenter_name,
        project_path=project_path,
        video_path=video_path,
        images_folder_path=images_folder_path,
        body_parts=body_parts,
        uniform_start=uniform_start,
        uniform_stop=uniform_stop,
        uniform_numframes=uniform_numframes,
        dummy_video_filename=dummy_video_filename,
        dummy_video_fps=dummy_video_fps
    )

    if cfg:
        print("Final config file path:", cfg)
    else:
        print("Project creation failed.")

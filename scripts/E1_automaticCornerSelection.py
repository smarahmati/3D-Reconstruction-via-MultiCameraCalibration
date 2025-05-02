import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv  # <-- for CSV writing

# ===============================================
# Global display size limit for ROI and preview
# ===============================================
MAX_DISPLAY_WIDTH  = 2500 # Ex: 1280 or 1920
MAX_DISPLAY_HEIGHT = 900 # Ex: 720 or 1080

VALID_IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def compute_scale_for_display(img, max_w=1280, max_h=720):
    """
    Compute a scale factor so the image fits within max_w x max_h.
    Returns scale <= 1.0. If the image is smaller, scale=1.0 (no change).
    """
    h_org, w_org = img.shape[:2]
    return min(1.0, max_w / float(w_org), max_h / float(h_org))

def selectROI_autoscale(window_name, full_img, max_width=1280, max_height=720):
    """
    Scale down if needed for display => cv2.selectROI => map ROI back to original coords.
    """
    h_org, w_org = full_img.shape[:2]
    scale = compute_scale_for_display(full_img, max_width, max_height)

    if scale < 1.0:
        new_w = int(w_org * scale)
        new_h = int(h_org * scale)
        resized_img = cv2.resize(full_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        r = cv2.selectROI(window_name, resized_img, showCrosshair=True)
        cv2.destroyWindow(window_name)
        x_scaled, y_scaled, w_scaled, h_scaled = r
        x_full = int(x_scaled / scale)
        y_full = int(y_scaled / scale)
        w_full = int(w_scaled / scale)
        h_full = int(h_scaled / scale)
        return (x_full, y_full, w_full, h_full)
    else:
        r = cv2.selectROI(window_name, full_img, showCrosshair=True)
        cv2.destroyWindow(window_name)
        return tuple(map(int, r))

def detect_20_corners_in_crop(full_img, roi, expected_corners=20):
    """
    Crop ROI => findChessboardCorners(5,4). If found => refine => shift coords to full-image.
    Return Nx2 corners if success, else None.
    """
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return None

    crop = full_img[y:y+h, x:x+w]
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray_crop, (5,4), None)
    if not found or corners.shape[0] != expected_corners:
        return None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
    corners = cv2.cornerSubPix(gray_crop, corners, (11,11), (-1,-1), criteria)

    corners_full = corners.copy()
    corners_full[:,0,0] += x
    corners_full[:,0,1] += y
    return corners_full

def show_corners_sequence_autoscale(full_img, corners_full,
                                    window_name="Preview",
                                    max_width=1280, max_height=720):
    """
    Draw corners in full_img => scale for display => prompt user 'y' to accept or else reject.
    Return True if accepted, False otherwise.
    """
    h_org, w_org = full_img.shape[:2]
    preview = full_img.copy()

    # Draw corner circles & connecting lines
    pts_line = corners_full.reshape(-1,2).astype(int)
    for i, (cx, cy) in enumerate(pts_line):
        cv2.circle(preview, (cx, cy), 5, (0,0,255), -1)
        if i < len(pts_line) - 1:
            cx2, cy2 = pts_line[i+1]
            cv2.line(preview, (cx, cy), (cx2, cy2), (255,0,0), 2)

    text_msg = "Press 'y' to ACCEPT, or any other key to REJECT"
    cv2.putText(preview, text_msg, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    scale = compute_scale_for_display(preview, max_width, max_height)
    if scale < 1.0:
        disp_w = int(w_org * scale)
        disp_h = int(h_org * scale)
        display_img = cv2.resize(preview, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    else:
        display_img = preview

    cv2.imshow(window_name, display_img)
    key = cv2.waitKey(0)
    cv2.destroyWindow(window_name)
    return (key in [ord('y'), ord('Y')])

def save_data_to_npz(npz_filename, corners_dict, resolution_dict, lab_reference_pose):
    """
    Save corners, resolutions, and the chosen reference pose in a .npz file.
    """
    keys_list = list(corners_dict.keys())
    corners_list = [corners_dict[k] for k in keys_list]
    resolutions_list = [resolution_dict[k] for k in keys_list]

    np.savez_compressed(
        npz_filename,
        corners_keys=keys_list,
        corners_vals=corners_list,
        resolutions_vals=resolutions_list,
        lab_reference_pose=np.array(lab_reference_pose, dtype=object)
    )
    print(f"[INFO] Saved corners, resolutions, and ref pose '{lab_reference_pose}' to {npz_filename}")

def save_data_to_csv(csv_filename, corners_dict, resolution_dict, lab_reference_pose):
    """
    Write out the same data that goes into the .npz, but in CSV format.
    One row per (pose_key, cam_str), plus a header line showing corner columns.
    """
    n_corners = 20

    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["# lab_reference_pose", lab_reference_pose])

        # Build header line
        header = ["pose_key", "cam_str", "img_w", "img_h"]
        for i in range(n_corners):
            header.append(f"corner_{i}_x")
            header.append(f"corner_{i}_y")
        writer.writerow(header)

        for (pose_key, cam_str) in corners_dict.keys():
            row = [pose_key, cam_str]
            (w_img, h_img) = resolution_dict[(pose_key, cam_str)]
            row.append(w_img)
            row.append(h_img)

            corners = corners_dict[(pose_key, cam_str)]  # shape (20,2)
            for i in range(n_corners):
                x_i = corners[i,0]
                y_i = corners[i,1]
                row.append(f"{x_i:.3f}")
                row.append(f"{y_i:.3f}")

            writer.writerow(row)

    print(f"[INFO] Saved corners data + reference pose info to CSV => {csv_filename}")


def main():
    # --------------------------------------------------------------------
    # 1) Prompt user for folder with chessboard images
    # --------------------------------------------------------------------
    image_folder = input("Enter path to folder with images: ").strip()
    if not os.path.isdir(image_folder):
        print(f"[ERROR] Invalid folder: {image_folder}")
        return

    # --------------------------------------------------------------------
    # 2) Collect all image files in that folder with known valid extensions
    # --------------------------------------------------------------------
    valid_exts = VALID_IMAGE_EXTS  # ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    all_files = [f for f in os.listdir(image_folder)
                 if f.lower().endswith(valid_exts)]
    all_files.sort()

    if not all_files:
        print(f"[ERROR] No image found in {image_folder} with extensions {valid_exts}")
        return

    # Build map: (pose_key, cam_str) => file path
    image_map = {}
    for fname in all_files:
        parts = fname.split('_')
        if len(parts) < 2:
            print(f"[WARN] Skipping {fname} (no underscore pattern)")
            continue
        cam_str  = parts[0]
        pose_key = parts[1]
        fpath    = os.path.join(image_folder, fname)
        image_map[(pose_key, cam_str)] = fpath

    print(f"[INFO] Found {len(image_map)} pose/cam matches in total.")

    # Identify unique poses & cameras
    unique_poses = sorted(set(k[0] for k in image_map.keys()))
    unique_cams  = sorted(set(k[1] for k in image_map.keys()))
    n_poses   = len(unique_poses)
    m_cameras = len(unique_cams)

    print("[INFO] Unique poses:", unique_poses)
    print("[INFO] Unique cams:", unique_cams)
    if n_poses == 0 or m_cameras == 0:
        print("[ERROR] No valid poses or cameras found in filenames (underscore pattern).")
        return

    # --------------------------------------------------------------------
    # 2A) Option: ask if user wants to see a mosaic of calibration images
    # --------------------------------------------------------------------
    user_mosaic_query = input("Do you want to see a mosaic pic of calibration images? (y/n): ").strip().lower()
    if user_mosaic_query in ["y","yes"]:
        # create mosaic
        fig, axes = plt.subplots(n_poses, m_cameras, figsize=(4*m_cameras,3*n_poses))
        if n_poses == 1 and m_cameras == 1:
            axes = np.array([[axes]])  # handle single subplot

        for r, pose in enumerate(unique_poses):
            for c, cam in enumerate(unique_cams):
                ax = axes[r,c] if (n_poses>1 or m_cameras>1) else axes
                ax.axis('off')
                title_str = f"{cam}_{pose}"
                if (pose, cam) in image_map:
                    fpath = image_map[(pose, cam)]
                    img = cv2.imread(fpath, cv2.IMREAD_COLOR)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        ax.imshow(img_rgb)
                    else:
                        ax.text(0.5,0.5, f"Could not read\n{title_str}", ha='center', va='center')
                else:
                    ax.text(0.5,0.5, f"Missing\n{title_str}", ha='center', va='center')
                ax.set_title(title_str)

        plt.tight_layout()

        # Save mosaic image
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()

        mosaic_path = os.path.join(script_dir, "MosaicOfCalImages.png")
        plt.savefig(mosaic_path)
        plt.close(fig)
        print(f"[INFO] Created mosaic => {mosaic_path}")
        print("Please open 'MosaicOfCalImages.png' manually if you want to review.\n")
    else:
        print("[INFO] Skipping mosaic creation.")

    # --------------------------------------------------------------------
    # 2B) Option: ask if user wants a lab reference pose
    # --------------------------------------------------------------------
    user_ref_query = input("Do you want to specify a lab reference pose? (y/n): ").strip().lower()

    ref_pose = None
    if user_ref_query in ['y','yes']:
        print("\nExamples: Enter '0' for the first pose, or a pose name like 'calpic01'...\n")
        user_input = input("Enter the lab reference pose (index or name): ").strip()
        try:
            idx = int(user_input)
            if 0 <= idx < len(unique_poses):
                ref_pose = unique_poses[idx]
                print(f"[INFO] Using pose '{ref_pose}' (by index).")
            else:
                print("[WARN] Index out of range => defaulting to first pose.")
                ref_pose = unique_poses[0]
        except ValueError:
            # parse as pose name
            if user_input in unique_poses:
                ref_pose = user_input
                print(f"[INFO] Using pose '{ref_pose}' (by name).")
            else:
                print("[WARN] Could not find that pose => default to first pose.")
                ref_pose = unique_poses[0]

        print(f"[INFO] Final chosen lab reference pose => '{ref_pose}'.")
    else:
        ref_pose = None
        print("[INFO] No reference pose => None.")

    # --------------------------------------------------------------------
    # 3) ROI selection + corner detection
    # --------------------------------------------------------------------
    N_EXPECTED = 20
    corners_dict    = {}
    resolution_dict = {}

    for pose_key in unique_poses:
        for cam_str in unique_cams:
            if (pose_key, cam_str) not in image_map:
                continue

            fpath = image_map[(pose_key, cam_str)]
            full_img = cv2.imread(fpath)
            if full_img is None:
                print(f"[WARN] Could not read {fpath}, skipping.")
                continue

            window_name = f"Select Board: {cam_str}_{pose_key}"
            roi = selectROI_autoscale(window_name, full_img,
                                      max_width=MAX_DISPLAY_WIDTH,
                                      max_height=MAX_DISPLAY_HEIGHT)
            x, y, w, h = roi
            if w <= 0 or h <= 0:
                print(f"[WARN] No valid ROI => {cam_str}_{pose_key}, skipping.")
                continue

            corners_full = detect_20_corners_in_crop(full_img, roi, N_EXPECTED)
            if corners_full is None:
                print(f"[WARN] Did not find exactly {N_EXPECTED} corners => {cam_str}_{pose_key}.")
                continue

            title_preview = f"Detected corners: {cam_str}_{pose_key}"
            accepted = show_corners_sequence_autoscale(
                full_img, corners_full,
                window_name=title_preview,
                max_width=MAX_DISPLAY_WIDTH,
                max_height=MAX_DISPLAY_HEIGHT
            )
            if accepted:
                corners_dict[(pose_key, cam_str)] = corners_full[:,0,:]
                h_img, w_img = full_img.shape[:2]
                resolution_dict[(pose_key, cam_str)] = (w_img, h_img)
                print(f"[ACCEPT] {pose_key} {cam_str}")
            else:
                print(f"[REJECT] {pose_key} {cam_str}")

    # --------------------------------------------------------------------
    # 4) Save .npz/.csv (including ref_pose if any)
    # --------------------------------------------------------------------
    npz_filename = "acceptedDetections_20Corners.npz"
    csv_filename = "acceptedDetections_20Corners.csv"
    save_data_to_npz(npz_filename, corners_dict, resolution_dict, lab_reference_pose=ref_pose)
    save_data_to_csv(csv_filename, corners_dict, resolution_dict, lab_reference_pose=ref_pose)
    print("[INFO] Done saving .npz + .csv corner data.")

    # --------------------------------------------------------------------
    # 5) OPTIONAL: Label images
    # --------------------------------------------------------------------
    for pose_key in unique_poses:
        for cam_str in unique_cams:
            key = (pose_key, cam_str)
            if key not in corners_dict:
                continue

            image_path = image_map[key]
            img = cv2.imread(image_path)
            if img is None:
                print(f"[WARN] Could not reload => {image_path} for labeling.")
                continue

            corners_2d = corners_dict[key]  # shape (20,2)
            for i in range(corners_2d.shape[0]):
                px = int(corners_2d[i,0])
                py = int(corners_2d[i,1])
                cv2.circle(img, (px, py), 4, (0,0,255), -1)

                # row,col => (i//5, i%5)
                row = i//5
                col = i%5

                # Label 1 => (row,col) near bottom-right
                label_rc = f"({row},{col})"
                cv2.putText(img, label_rc, (px+0, py+10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)

                # Label 2 => corner index i => top-left
                label_i = f"#{i}"
                cv2.putText(img, label_i, (px-0, py-10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)

            if pose_key == ref_pose:
                labeled_fname = f"RefPose_{pose_key}_{cam_str}_labeled.png"
            else:
                labeled_fname = f"Pose_{pose_key}_{cam_str}_labeled.png"
            cv2.imwrite(labeled_fname, img)
            print(f"[INFO] Saved labeled => {labeled_fname}")

    print("[DONE] Processing finished.")

if __name__=="__main__":
    main()

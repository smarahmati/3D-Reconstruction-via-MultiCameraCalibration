import os
import cv2  # for reading original video's FPS/frame count
import csv
import math
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

##############################################################################
# User Settings & Main Inputs
##############################################################################

CSV_3D_FILE     = "All3DMarkers.csv"
REAL_VIDEO_PATH = "Cam2_2025-04-21 22-29-36.mp4"
AXIS_MARGIN     = 10
MARKER_SIZE     = 80
PLOT_TITLE      = "3D Markers Animation"

VIDEO_SIZE      = (1920, 1080)
DPI             = 100

ANIMATION_OUTFILE        = "MarkersAnimation.mp4"
FFMPEG_EXECUTABLE_PATH   = r"C:\FFmpeg\bin\ffmpeg.exe"  # <--- Your ffmpeg.exe path

def main():
    # 1) Configure Matplotlib so it knows ffmpeg's location:
    matplotlib.rcParams["animation.ffmpeg_path"] = FFMPEG_EXECUTABLE_PATH

    # 2) Read the original video => total time
    if not os.path.isfile(REAL_VIDEO_PATH):
        print(f"[ERROR] Could not find real video => {REAL_VIDEO_PATH}")
        return
    cap = cv2.VideoCapture(REAL_VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open => {REAL_VIDEO_PATH}")
        return

    real_fps = cap.get(cv2.CAP_PROP_FPS)
    real_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if real_fps <= 0 or real_frame_count < 1:
        print("[ERROR] The real video has invalid FPS or frame count.")
        return

    real_video_duration = real_frame_count / real_fps
    print(f"[INFO] Original video => {real_frame_count} frames at {real_fps:.2f} FPS => total {real_video_duration:.2f}s")

    # 3) Parse 3D CSV
    if not os.path.isfile(CSV_3D_FILE):
        print(f"[ERROR] Could not find 3D CSV => {CSV_3D_FILE}")
        return

    with open(CSV_3D_FILE, 'r', newline='') as f:
        rows = list(csv.reader(f))
    if len(rows)<2:
        print("[ERROR] 3D CSV has <2 lines => no data.")
        return

    header = rows[0]
    data_rows = rows[1:]

    marker_col_map = {}
    for c in range(1, len(header)):
        colName = header[c].strip()
        if colName.endswith("_X") or colName.endswith("_Y") or colName.endswith("_Z"):
            base = colName[:-2]  # e.g. "P01"
            axis = colName[-1]   # 'X','Y','Z'
            if base not in marker_col_map:
                marker_col_map[base] = {"X":None,"Y":None,"Z":None}
            marker_col_map[base][axis] = c

    marker_cols = {}
    for mName, dct in marker_col_map.items():
        if dct["X"] is not None and dct["Y"] is not None and dct["Z"] is not None:
            marker_cols[mName] = (dct["X"], dct["Y"], dct["Z"])

    marker_names = sorted(marker_cols.keys())
    if not marker_names:
        print("[ERROR] No valid marker columns => cannot proceed.")
        return

    markerCount = len(marker_names)
    frameCount  = len(data_rows)
    print(f"[INFO] 3D CSV => {frameCount} frames, {markerCount} markers => {marker_names}")

    all_3d = np.full((frameCount, markerCount, 3), np.nan, dtype=float)
    frame_nums = np.zeros(frameCount, dtype=int)

    for i, row in enumerate(data_rows):
        try:
            frm = int(row[0])
        except:
            frm = i
        frame_nums[i] = frm

        for mIdx, mName in enumerate(marker_names):
            (cX,cY,cZ) = marker_cols[mName]
            xStr = row[cX].strip() if cX<len(row) else ""
            yStr = row[cY].strip() if cY<len(row) else ""
            zStr = row[cZ].strip() if cZ<len(row) else ""
            if xStr and yStr and zStr:
                try:
                    xx = float(xStr)
                    yy = float(yStr)
                    zz = float(zStr)
                    all_3d[i,mIdx,0] = xx
                    all_3d[i,mIdx,1] = yy
                    all_3d[i,mIdx,2] = zz
                except:
                    pass

    valid_mask = ~np.isnan(all_3d)
    valid_count = np.count_nonzero(valid_mask)
    if valid_count==0:
        print("[ERROR] All entries are NaN => no numeric data found.")
        return
    print(f"[INFO] Found {valid_count} numeric entries (X,Y,Z).")

    # 4) Auto-limits
    Xvals = all_3d[:,:,0].flatten()
    Yvals = all_3d[:,:,1].flatten()
    Zvals = all_3d[:,:,2].flatten()

    Xvals = Xvals[~np.isnan(Xvals)]
    Yvals = Yvals[~np.isnan(Yvals)]
    Zvals = Zvals[~np.isnan(Zvals)]

    minX, maxX = Xvals.min(), Xvals.max()
    minY, maxY = Yvals.min(), Yvals.max()
    minZ, maxZ = Zvals.min(), Zvals.max()

    ax_min_x = minX - AXIS_MARGIN
    ax_max_x = maxX + AXIS_MARGIN
    ax_min_y = minY - AXIS_MARGIN
    ax_max_y = maxY + AXIS_MARGIN
    ax_min_z = minZ - AXIS_MARGIN
    ax_max_z = maxZ + AXIS_MARGIN

    print("[INFO] Auto axis range =>")
    print(f"   X: [{ax_min_x:.3f}, {ax_max_x:.3f}]")
    print(f"   Y: [{ax_min_y:.3f}, {ax_max_y:.3f}]")
    print(f"   Z: [{ax_min_z:.3f}, {ax_max_z:.3f}]")

    # 5) Build figure
    figW_inch = VIDEO_SIZE[0]/float(DPI)
    figH_inch = VIDEO_SIZE[1]/float(DPI)
    fig = plt.figure(figsize=(figW_inch, figH_inch), dpi=DPI)
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title(PLOT_TITLE)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    ax.set_xlim(ax_min_x, ax_max_x)
    ax.set_ylim(ax_min_y, ax_max_y)
    ax.set_zlim(ax_min_z, ax_max_z)

    scatter_points = ax.scatter([], [], [], s=MARKER_SIZE, c='blue')

    # 6) total anim time = real_video_duration => so anim_fps = frameCount / real_video_duration
    cap_fps = float(real_fps)
    cap_frames = float(real_frame_count)
    total_time = cap_frames/cap_fps if cap_fps>0 else 1.0
    anim_fps = frameCount / total_time
    anim_interval_ms = 1000.0 / anim_fps

    print(f"[INFO] We'll produce {frameCount} frames at ~{anim_fps:.2f} FPS => {total_time:.2f}s total")

    def init_anim():
        scatter_points._offsets3d = ([],[],[])
        return (scatter_points,)

    def update_anim(frameIdx):
        xdata = all_3d[frameIdx,:,0]
        ydata = all_3d[frameIdx,:,1]
        zdata = all_3d[frameIdx,:,2]
        mask  = ~np.isnan(xdata)
        xdata = xdata[mask]
        ydata = ydata[mask]
        zdata = zdata[mask]
        scatter_points._offsets3d = (xdata,ydata,zdata)

        # show user which frame
        fnum = frame_nums[frameIdx]
        ax.set_title(f"{PLOT_TITLE} - Frame {fnum}")
        return (scatter_points,)

    ani = animation.FuncAnimation(
        fig,
        update_anim,
        frames=frameCount,
        init_func=init_anim,
        interval=anim_interval_ms,
        blit=False
    )

    # 7) Save MP4 => using the user-defined ffmpeg path
    writer_fps = float(anim_fps)
    ffWriter = animation.FFMpegWriter(fps=writer_fps)  # do not pass executable=...
    # Instead, we've set rcParams above => matplotlib.rcParams["animation.ffmpeg_path"] = FFMPEG_EXECUTABLE_PATH

    print(f"[INFO] Saving => {ANIMATION_OUTFILE} with custom ffmpeg path")
    ani.save(ANIMATION_OUTFILE, writer=ffWriter)
    print(f"[INFO] Saved => {ANIMATION_OUTFILE}")

    # 8) show
    plt.show(block=True)
    input("Press ENTER to quit...")

if __name__=="__main__":
    main()

import os
import glob
import csv
import cv2
import numpy as np
from collections import defaultdict

##############################################################################
# User Settings & Main Inputs
##############################################################################
SYSTEM_PARAMS_FILE = "SystemParameters.npz"  # e.g. from your calibration code
DLC_FOLDER = "DLC_csv_folder"  # folder path containing the DeepLabCut CSV files
OUT_2D_CSV = "All2D_data.csv"  # name for combined 2D output
OUT_3D_CSV = "All3DMarkers.csv"  # name for final 3D output


def main():
    # 1) Load system parameters (intrinsics + extrinsics) from NPZ
    if not os.path.isfile(SYSTEM_PARAMS_FILE):
        print(f"[ERROR] Could not find system params file: {SYSTEM_PARAMS_FILE}")
        return

    data_npz = np.load(SYSTEM_PARAMS_FILE, allow_pickle=True)
    camera_intrinsics = data_npz["camera_intrinsics"].item()  # {camStr: (K, distCoeffs)}
    camera_extrinsics = data_npz["camera_extrinsics"].item()  # {camStr: (R_gc, t_gc)}
    reference_pose = data_npz["lab_reference_pose"].item()  # e.g. 'calpic01' or None

    cam_list = sorted(camera_intrinsics.keys())
    print("[INFO] Cameras from system params:", cam_list)
    print("[INFO] Reference pose in system params =>", reference_pose)

    # 2) Collect all .csv from the DLC folder
    if not os.path.isdir(DLC_FOLDER):
        print(f"[ERROR] Not a valid folder: {DLC_FOLDER}")
        return

    all_dlc_csv = sorted(glob.glob(os.path.join(DLC_FOLDER, "*.csv")))
    if not all_dlc_csv:
        print(f"[ERROR] No CSV found in {DLC_FOLDER}")
        return

    # We'll parse them => build frame2d[frameIdx][markerName][camStr] = (x, y)
    frame2d = defaultdict(lambda: defaultdict(dict))
    marker_names_set = set()

    # parse each CSV => determine camera label from filename e.g. "Cam2_"
    for csvfile in all_dlc_csv:
        fname = os.path.basename(csvfile)
        # find which camera => e.g. if cstr='Cam2' => search "Cam2_" in fname
        camera_label = None
        for cstr in cam_list:
            if cstr + "_" in fname:
                camera_label = cstr
                break

        if camera_label is None:
            print(f"[WARN] Could not match any known camera label for {fname}, skipping.")
            continue

        print(f"[INFO] Parsing DLC CSV for camera: {camera_label} => {csvfile}")
        with open(csvfile, 'r', newline='') as f:
            reader = csv.reader(f)
            all_rows = list(reader)

        # We'll attempt to find which rows/cols correspond to "x" or "y"
        # e.g. if cell at row i, col c => 'x' => the marker name is in row i-1, col c
        # and the data is row i+1 onward. We'll handle partial cases carefully.

        # We'll define a structure: column_map[c] = (markerName, "x" or "y", dataStartRowIndex)
        column_map = {}  # colIndex => (markerName, 'x'/'y', dataStartRow)
        # We'll search from row 1..(len(all_rows)-1) because we skip i=0 (can't go i-1)
        # and i+1 must exist for data
        rowCount = len(all_rows)
        for i in range(1, rowCount - 1):
            row_i = all_rows[i]
            row_i_1 = all_rows[i - 1]  # the row above
            # We'll parse columns up to the min length
            colCount = len(row_i)
            for c in range(colCount):
                cellVal = row_i[c].strip().lower()
                if cellVal in ["x", "y"] and cellVal not in ["likelihood"]:
                    # the marker name is row_i_1[c]
                    # data is from row i+1 onward
                    if c < len(row_i_1):
                        markerName = row_i_1[c].strip()
                        if markerName != "":
                            # store
                            column_map[(i, c)] = (markerName, cellVal, i + 1)

        if not column_map:
            print(f"[WARN] Could not find any 'x'/'y' columns in {csvfile}, skipping.")
            continue

        # Now we parse the data for each column_map entry
        # for each => from row dataStartRow..end => rowData for frames
        for ((headerRow, colIndex), (markerName, xyCode, dataStartRow)) in column_map.items():
            # we read from dataStartRow..end
            for rData in range(dataStartRow, rowCount):
                rowData = all_rows[rData]
                if colIndex >= len(rowData):
                    continue
                cellStr = rowData[colIndex].strip()
                if cellStr == "":
                    continue
                # The first column might be frame index
                # Let's try parse rowData[0] as frame
                try:
                    frameNumber = int(rowData[0])
                except:
                    frameNumber = rData - dataStartRow

                try:
                    valF = float(cellStr)
                except:
                    continue

                # store => frame2d[frameNumber][markerName][camera_label]["x"/"y"] = valF
                # but we keep them in subdict => then we unify x,y if both appear
                if markerName not in frame2d[frameNumber]:
                    frame2d[frameNumber][markerName][camera_label] = {"x": None, "y": None}
                # ensure subdict
                if camera_label not in frame2d[frameNumber][markerName]:
                    frame2d[frameNumber][markerName][camera_label] = {"x": None, "y": None}

                frame2d[frameNumber][markerName][camera_label][xyCode] = valF
                marker_names_set.add(markerName)

    marker_names = sorted(marker_names_set)
    print("[INFO] Found markers:", marker_names)

    # 3) Write combined 2D => columns => frame, marker_camX_X, marker_camX_Y, ...
    with open(OUT_2D_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["frame"]
        for M in marker_names:
            for c in cam_list:
                header.append(f"{M}_{c}_X")
                header.append(f"{M}_{c}_Y")
        writer.writerow(header)

        allFrames = sorted(frame2d.keys())
        for fr in allFrames:
            rowOut = [fr]
            for M in marker_names:
                for c in cam_list:
                    if c in frame2d[fr][M]:
                        xVal = frame2d[fr][M][c].get("x", None)
                        yVal = frame2d[fr][M][c].get("y", None)
                        if xVal is not None:
                            rowOut.append(f"{xVal:.3f}")
                        else:
                            rowOut.append("")
                        if yVal is not None:
                            rowOut.append(f"{yVal:.3f}")
                        else:
                            rowOut.append("")
                    else:
                        rowOut.append("")
                        rowOut.append("")
            writer.writerow(rowOut)

    print(f"[INFO] Wrote combined 2D CSV => {OUT_2D_CSV}")

    # 4) Triangulate => for each frame, each marker => gather cameras => pairwise => average
    final3d = defaultdict(dict)

    def triangulate_pairwise(camA, camB, xA, yA, xB, yB):
        # Ensure camA, camB in intrinsics + extrinsics
        if (camA not in camera_intrinsics) or (camA not in camera_extrinsics):
            return None
        if (camB not in camera_intrinsics) or (camB not in camera_extrinsics):
            return None
        (K_A, distA) = camera_intrinsics[camA]
        (R_gA, t_gA) = camera_extrinsics[camA]
        (K_B, distB) = camera_intrinsics[camB]
        (R_gB, t_gB) = camera_extrinsics[camB]

        ptA = np.array([xA, yA, 1], dtype=float).reshape(3, 1)
        ptB = np.array([xB, yB, 1], dtype=float).reshape(3, 1)

        P_A = K_A @ np.hstack([R_gA, t_gA])
        P_B = K_B @ np.hstack([R_gB, t_gB])

        X_hom = cv2.triangulatePoints(P_A, P_B, ptA[:2], ptB[:2])
        if abs(X_hom[3]) < 1e-12:
            return None
        return (X_hom[:3] / X_hom[3]).flatten()

    for fr in sorted(frame2d.keys()):
        for M in marker_names:
            # gather cameras that have x,y
            cPoints = []
            for c in cam_list:
                if c in frame2d[fr][M]:
                    xVal = frame2d[fr][M][c].get("x", None)
                    yVal = frame2d[fr][M][c].get("y", None)
                    if (xVal is not None) and (yVal is not None):
                        cPoints.append((c, xVal, yVal))
            if len(cPoints) < 2:
                continue

            tri3dList = []
            for i in range(len(cPoints)):
                for j in range(i + 1, len(cPoints)):
                    (camA, xA, yA) = cPoints[i]
                    (camB, xB, yB) = cPoints[j]
                    X_3d = triangulate_pairwise(camA, camB, xA, yA, xB, yB)
                    if X_3d is not None:
                        tri3dList.append(X_3d)
            if len(tri3dList) == 0:
                continue
            arr = np.array(tri3dList)
            mean3d = arr.mean(axis=0)
            final3d[fr][M] = mean3d

    # 5) Write final 3D => columns => frame, markerX, markerY, markerZ
    with open(OUT_3D_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["frame"]
        for M in marker_names:
            header.append(f"{M}_X")
            header.append(f"{M}_Y")
            header.append(f"{M}_Z")
        writer.writerow(header)

        allFrames3D = sorted(final3d.keys())
        for fr in allFrames3D:
            rowOut = [fr]
            for M in marker_names:
                if M in final3d[fr]:
                    (Xx, Yy, Zz) = final3d[fr][M]
                    rowOut.append(f"{Xx:.3f}")
                    rowOut.append(f"{Yy:.3f}")
                    rowOut.append(f"{Zz:.3f}")
                else:
                    rowOut.append("")
                    rowOut.append("")
                    rowOut.append("")
            writer.writerow(rowOut)

    print(f"[INFO] Wrote final 3D CSV => {OUT_3D_CSV}")
    print("[DONE] 3D reconstruction completed.")


if __name__ == "__main__":
    main()

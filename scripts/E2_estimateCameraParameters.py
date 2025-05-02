import os
import numpy as np
import cv2
import itertools
from collections import defaultdict
import math

import plotly.graph_objects as go
import plotly.io as pio  # for writing/auto-opening HTML

# Just like before, plus new STL references:
from stl import mesh

##############################################################################
# User Settings & Main Inputs
##############################################################################
NPZ_FILENAME    = "acceptedDetections_20Corners.npz"
SQUARE_SIZE_MM  = 35.0      # physical size of each chessboard square, in mm
CHESSBOARD_SIZE = (5,4)     # pattern size: 5 corners in x-direction, 4 in y-direction

ORIGIN_CORNER_INDEX = 15     # Which corner index becomes (0,0,0)
XAXIS_PAIR          = (15,16)    # Which corners define +X direction
ZAXIS_PAIR          = (15,10)    # Which corners define +Z direction

# NEW: STL file info
STL_FILE      = "camera.stl"          # path to your camera .stl
STL_CENTER    = [18.35, 8.95, 10.1]   # local coords in STL that define the "center"
CAMERA_SCALE  = 3.0                  # user scale factor

##############################################################################
# (Your same helper functions) build_chessboard_3dpoints, custom_transform_corners, ...
##############################################################################

def build_chessboard_3dpoints(pattern_size, square_size):
    """
    First build a standard "row-major" array of corners in the plane.
    5 wide x 4 tall => corners indexed row-major:
      index i => row=i//5, col=i%5
    We'll place the board in the x-z plane (y=0) by default,
    with +X to the right, +Z up.
    """
    w, h = pattern_size
    pts_3d = []
    for j in range(h):        # j => row
        for i in range(w):    # i => col
            # row = j, col = i
            x_3d = i * square_size
            y_3d = 0.0
            z_3d = j * square_size
            pts_3d.append((x_3d, y_3d, z_3d))
    return np.array(pts_3d, dtype=np.float32)  # shape (N, 3)

def custom_transform_corners(std_corners, origin_idx, xaxis_pair, zaxis_pair):
    """
    Takes the standard row-major corners (N,3), plus user-specified corner indices:
      - origin_idx => this corner => (0,0,0)
      - xaxis_pair => (i1,i2) => define +X direction from corner i1 to corner i2
      - zaxis_pair => (j1,j2) => define +Z direction from corner j1 to corner j2

    Returns a new array (N,3) in the custom reference frame:
      - corner at origin_idx is now 0,0,0
      - The direction i2-i1 => +X
      - The direction j2-j1 => +Z
      - The Y axis => Z cross X (to ensure right-handed).
    """

    # 1) shift so that origin_idx is at (0,0,0)
    origin_3d   = std_corners[origin_idx]            # shape (3,)
    shifted_pts = std_corners - origin_3d            # (N,3)

    # 2) define local X, Z from the difference of corners
    xA = std_corners[xaxis_pair[0]]
    xB = std_corners[xaxis_pair[1]]
    # but also shift them by origin_3d
    vX = (xB - origin_3d) - (xA - origin_3d)  # or simply (xB - xA).
    # In practice, let's do xB - xA directly:
    vX = std_corners[xaxis_pair[1]] - std_corners[xaxis_pair[0]]

    zA = std_corners[zaxis_pair[0]]
    zB = std_corners[zaxis_pair[1]]
    vZ = std_corners[zaxis_pair[1]] - std_corners[zaxis_pair[0]]

    # 3) normalize them
    # In case user picks the same corners or a very short vector => handle errors
    eps = 1e-12
    normX = np.linalg.norm(vX)
    normZ = np.linalg.norm(vZ)
    if normX < eps or normZ < eps:
        print("[WARN] X-axis or Z-axis pair define zero-length vector. Check your corner indices!")
        vX = np.array([1.0, 0.0, 0.0], dtype=float)
        vZ = np.array([0.0, 0.0, 1.0], dtype=float)
        normX, normZ = 1, 1
    else:
        vX = vX / normX
        vZ = vZ / normZ

    # 4) Ensure right-handed frame: we define Y = Z cross X, then re-orthonormalize
    vY = np.cross(vZ, vX)  # Z cross X => Y
    normY = np.linalg.norm(vY)
    if normY < eps:
        print("[WARN] The chosen X-axis and Z-axis vectors are nearly collinear.")
        # fallback
        vY = np.array([0.0, 1.0, 0.0], dtype=float)
        normY = 1
    else:
        vY = vY / normY

    # Recompute Z = X cross Y to ensure perfect orthonormal frame
    vZ = np.cross(vX, vY)
    # final check
    vZ_norm = np.linalg.norm(vZ)
    if vZ_norm < eps:
        # fallback
        vZ = np.array([0,0,1], dtype=float)
    else:
        vZ = vZ / vZ_norm

    # So we have X= vX, Y= vY, Z= vZ => 3x3 rotation
    R = np.stack([vX, vY, vZ], axis=1)  # shape (3,3)

    # 5) transform each point:
    # local_coord = R^T * shifted_coord
    # so newPts = (N,3)
    new_pts = shifted_pts @ R  # because R is (3,3) if we do (N,3) @ (3,3) => (N,3)

    return new_pts

def load_npz_data(npz_filename):
    data = np.load(npz_filename, allow_pickle=True)
    keys_list = data["corners_keys"]
    corners_list = data["corners_vals"]
    resolutions_list = data["resolutions_vals"]
    lab_reference_pose = data["lab_reference_pose"].item()  # stored as object dtype

    corners_dict = {}
    resolution_dict = {}
    for (k, c, r) in zip(keys_list, corners_list, resolutions_list):
        corners_dict[tuple(k)] = c
        resolution_dict[tuple(k)] = r

    return corners_dict, resolution_dict, lab_reference_pose

def collect_camera_data(corners_dict):
    from collections import defaultdict
    pose2cams = defaultdict(list)
    for (pose_key, cam_str) in corners_dict.keys():
        pose2cams[pose_key].append(cam_str)

    cam_set  = set()
    pose_set = set()
    for (pose_key, cam_str) in corners_dict.keys():
        cam_set.add(cam_str)
        pose_set.add(pose_key)

    return cam_set, pose_set, dict(pose2cams)

def calibrate_camera_intrinsics(cam_str, corners_dict, resolution_dict,
                                valid_poses, pattern_3d):
    objpoints = []
    imgpoints = []
    img_size  = None

    for pose_key in valid_poses:
        if (pose_key, cam_str) not in corners_dict:
            continue
        corners_2d = corners_dict[(pose_key, cam_str)]
        w_img, h_img = resolution_dict[(pose_key, cam_str)]
        img_size = (w_img, h_img)

        objpoints.append(pattern_3d)
        imgpoints.append(corners_2d.astype(np.float32))

    if len(objpoints) < 1:
        print(f"[WARN] Camera {cam_str} has no valid multi-corner poses. No calibration performed.")
        return None, None, None

    f_guess = max(img_size[0], img_size[1])
    cx = img_size[0] * 0.5
    cy = img_size[1] * 0.5
    init_camera_matrix = np.array([
        [f_guess, 0,      cx],
        [0,       f_guess, cy],
        [0,       0,      1  ]
    ], dtype=np.float64)
    distCoeffs_init = np.zeros(5, dtype=np.float64)

    flags = (cv2.CALIB_USE_INTRINSIC_GUESS |
             cv2.CALIB_FIX_ASPECT_RATIO)

    rms, K, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objectPoints=objpoints,
        imagePoints=imgpoints,
        imageSize=img_size,
        cameraMatrix=init_camera_matrix,
        distCoeffs=distCoeffs_init,
        flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    )

    return K, distCoeffs, rms

def solve_pnp_for_pose(pattern_3d, corners_2d, K, distCoeffs):
    success, rvec, tvec = cv2.solvePnP(
        pattern_3d,
        corners_2d.astype(np.float32),
        K, distCoeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    return R.astype(np.float32), tvec.reshape(3,1).astype(np.float32)

def invert_rt(R, t):
    R_inv = R.transpose()
    t_inv = -R_inv @ t
    return R_inv, t_inv

def compose_rt(R1, t1, R2, t2):
    R12 = R1 @ R2
    t12 = R1 @ t2 + t1
    return R12, t12

def rt_to_cam_center(R, t):
    """Camera center in global coords => -R^T * t"""
    return -R.transpose() @ t

def draw_camera_axes(fig, center_3d, R_gc, scale=30.0, cam_name="CamX"):
    R_cg = R_gc.transpose() # I did this modification to find how the local camera X/Y/Z axes appear in global coordinates
    x_dir_global = R_cg @ np.array([1,0,0], dtype=float)
    y_dir_global = R_cg @ np.array([0,1,0], dtype=float)
    z_dir_global = R_cg @ np.array([0,0,1], dtype=float)

    x_end = center_3d + scale*x_dir_global
    y_end = center_3d + scale*y_dir_global
    z_end = center_3d + scale*z_dir_global

    # X axis = red
    fig.add_trace(go.Scatter3d(
        x=[center_3d[0], x_end[0]],
        y=[center_3d[1], x_end[1]],
        z=[center_3d[2], x_end[2]],
        mode='lines',
        line=dict(color='red', width=3),
        name=f"{cam_name}_X"
    ))
    # Y axis = green
    fig.add_trace(go.Scatter3d(
        x=[center_3d[0], y_end[0]],
        y=[center_3d[1], y_end[1]],
        z=[center_3d[2], y_end[2]],
        mode='lines',
        line=dict(color='green', width=3),
        name=f"{cam_name}_Y"
    ))
    # Z axis = blue
    fig.add_trace(go.Scatter3d(
        x=[center_3d[0], z_end[0]],
        y=[center_3d[1], z_end[1]],
        z=[center_3d[2], z_end[2]],
        mode='lines',
        line=dict(color='blue', width=3),
        name=f"{cam_name}_Z"
    ))



def save_system_parameters_npz(out_filename, camera_intrinsics, camera_extrinsics, reference_pose):
    """
    camera_intrinsics: dict => {cam_str: (K, distCoeffs)}
    camera_extrinsics: dict => {cam_str: (R_gc, t_gc)} (global->camera)
    reference_pose   : e.g. 'calpic01' or None
    """
    # Convert dicts to something npz-friendly:
    cams = sorted(camera_intrinsics.keys())
    # Build arrays or sub-dicts
    all_intrinsics = {}
    all_extrinsics = {}
    for c in cams:
        (K, dist) = camera_intrinsics[c]
        (R, t)    = camera_extrinsics.get(c, (None, None))
        all_intrinsics[c] = (K, dist)
        if R is not None:
            all_extrinsics[c] = (R, t)
        else:
            all_extrinsics[c] = (None, None)

    np.savez_compressed(
        out_filename,
        camera_intrinsics=all_intrinsics,
        camera_extrinsics=all_extrinsics,
        lab_reference_pose=reference_pose
    )
    print(f"[INFO] Saved system parameters to {out_filename}")


def save_system_parameters_csv(out_csv, camera_intrinsics, camera_extrinsics, reference_pose):
    """
    Save camera intrinsics & extrinsics + reference pose in a CSV.
    One row per camera => each numeric value in its own column.
    Columns:
      cam_id,
      fx, fy, cx, cy,
      dist_0, dist_1, ... (as many as the length of distCoeffs),
      R_00, R_01, R_02,
      R_10, R_11, R_12,
      R_20, R_21, R_22,
      t_x, t_y, t_z
    """

    import csv
    cams = sorted(camera_intrinsics.keys())

    # Figure out the maximum number of distortion coefficients across all cams
    max_dist_len = 0
    for c in cams:
        (K, dist) = camera_intrinsics[c]
        dlen = dist.size if dist is not None else 0
        if dlen > max_dist_len:
            max_dist_len = dlen

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write the reference pose at the top
        writer.writerow(["#lab_reference_pose", reference_pose])

        # Build a flexible header
        header = [
            "cam_id",
            "fx","fy","cx","cy"
        ]

        # Add columns for each distortion coefficient
        for i in range(max_dist_len):
            header.append(f"dist_{i}")

        # Add columns for 9 R elements
        for rrow in range(3):
            for rcol in range(3):
                header.append(f"R_{rrow}{rcol}")

        # Then t_x, t_y, t_z
        header.append("t_x")
        header.append("t_y")
        header.append("t_z")

        writer.writerow(header)

        for c in cams:
            (K, dist) = camera_intrinsics[c]
            (R, t)    = camera_extrinsics.get(c, (None, None))
            if R is None:
                # Means no extrinsic
                R = np.eye(3,dtype=float)
                t = np.zeros((3,1),dtype=float)

            fx = K[0,0]
            fy = K[1,1]
            cx = K[0,2]
            cy = K[1,2]

            dist_array = dist.ravel() if dist is not None else np.zeros((max_dist_len,), dtype=float)
            # or pad with zeros if dist has fewer than max_dist_len

            # Build the row
            row = []
            row.append(c)  # cam_id
            row.append(f"{fx:.6f}")
            row.append(f"{fy:.6f}")
            row.append(f"{cx:.6f}")
            row.append(f"{cy:.6f}")

            # DistCoeffs
            # If dist has fewer than max_dist_len, fill the rest with 0 or 0.0
            dvals = []
            for i in range(max_dist_len):
                if i < len(dist_array):
                    dvals.append(f"{dist_array[i]:.6f}")
                else:
                    dvals.append("0.000000")
            row.extend(dvals)

            # R
            R_flat = R.flatten()
            for val in R_flat:
                row.append(f"{val:.6f}")

            # t
            tx, ty, tz = t.ravel()
            row.append(f"{tx:.6f}")
            row.append(f"{ty:.6f}")
            row.append(f"{tz:.6f}")

            writer.writerow(row)

    print(f"[INFO] Saved system parameters to CSV => {out_csv}")



##############################################################################
# NEW function to add the STL camera model
##############################################################################
def draw_stl_camera_model(
    fig,
    stl_path,
    stl_center_local,
    scale_factor,
    R_gc,
    t_gc,
    name="STL_CamModel"
):
    """
    stl_path: path to camera STL
    stl_center_local: (x,y,z) in STL coords that define the 'center' => we shift so that becomes (0,0,0)
    scale_factor: user scale => multiply each local vertex by this
    R_gc, t_gc => usual 'global->camera' rotation and translation
                  We'll invert => R_cg=R_gc.T => place the model in global coords
    Then we apply an extra 180° rotation around Z in local space so that
    STL +X => camera -X, STL +Y => camera -Y, STL +Z => camera +Z.
    """
    from stl import mesh
    import numpy as np
    import plotly.graph_objects as go

    # 1) Load STL => Nx3 triangles
    your_mesh = mesh.Mesh.from_file(stl_path)
    X = your_mesh.x.flatten()
    Y = your_mesh.y.flatten()
    Z = your_mesh.z.flatten()
    num_triangles = len(X)//3
    i_idx = np.array([3*t for t in range(num_triangles)])
    j_idx = np.array([3*t+1 for t in range(num_triangles)])
    k_idx = np.array([3*t+2 for t in range(num_triangles)])
    verts_local = np.vstack([X, Y, Z]).T  # shape (N,3)

    # 2) Shift => stl_center_local => (0,0,0)
    shift = np.array(stl_center_local, dtype=float)
    verts_local -= shift

    # 3) Scale
    verts_local *= scale_factor

    # 4) Rotate local axes 180° around Z => flip X,Y => diag([-1, -1, 1])
    flip_xy = np.diag([-1.0, -1.0, 1.0])  # shape (3,3)
    verts_local = verts_local @ flip_xy.T

    # 5) camera->global => R_cg= R_gc.T, plus center
    R_gc = np.array(R_gc, dtype=float)
    t_gc = np.array(t_gc, dtype=float).reshape(3,1)
    R_cg = R_gc.T
    cam_center = -R_cg @ t_gc
    cam_center = cam_center.flatten()

    # transform => v_global = (verts_local @ R_cg.T) + cam_center
    verts_global = (verts_local @ R_cg.T) + cam_center

    # 6) Add Mesh3d to Plotly
    fig.add_trace(go.Mesh3d(
        x=verts_global[:,0],
        y=verts_global[:,1],
        z=verts_global[:,2],
        i=i_idx,
        j=j_idx,
        k=k_idx,
        color='gray',
        opacity=0.5,
        name=name
    ))


##############################################################################
# Main
##############################################################################
def main():
    print("[INFO] Loading NPZ data...")
    corners_dict, resolution_dict, lab_ref_pose = load_npz_data(NPZ_FILENAME)

    cam_set, pose_set, pose2cams = collect_camera_data(corners_dict)
    cam_list  = sorted(cam_set)
    pose_list = sorted(pose_set)

    print(f"[INFO] Found {len(cam_list)} cameras: {cam_list}")
    print(f"[INFO] Found {len(pose_list)} poses: {pose_list}")
    print(f"[INFO] 'lab_reference_pose' from NPZ => {lab_ref_pose}")

    valid_poses = [p for p in pose_list if len(pose2cams[p]) >= 2]
    print(f"[INFO] Among them, {len(valid_poses)} have >= 2 cameras => {valid_poses}")

    std_pattern_3d = build_chessboard_3dpoints(CHESSBOARD_SIZE, SQUARE_SIZE_MM)
    pattern_3d_custom = custom_transform_corners(
        std_pattern_3d,
        origin_idx=ORIGIN_CORNER_INDEX,
        xaxis_pair=XAXIS_PAIR,
        zaxis_pair=ZAXIS_PAIR
    )

    # 1) Intrinsics
    camera_intrinsics = {}
    for cam_str in cam_list:
        K, distCoeffs, rms = calibrate_camera_intrinsics(cam_str, corners_dict, resolution_dict, valid_poses, pattern_3d_custom)
        if K is None:
            print(f"[WARN] No intrinsics for camera {cam_str}")
        else:
            camera_intrinsics[cam_str] = (K, distCoeffs)
            print(f"[INFO] Camera={cam_str}, RMS={rms:.4f}")
            print("       Intrinsic Matrix K=\n", K)
            print("       Distortion Coeffs:", distCoeffs.ravel())

    # 2) Solve extrinsics
    reference_pose = lab_ref_pose
    print(f"[INFO] The reference pose is: {reference_pose}")

    camera_extrinsics = {}
    def attempt_solve_extrinsics_for_pose(pose_key):
        from collections import defaultdict
        cams_in_pose = pose2cams[pose_key]
        board2cam={}
        for c_str in cams_in_pose:
            if c_str not in camera_intrinsics:
                continue
            corners_2d = corners_dict[(pose_key, c_str)]
            K, dist = camera_intrinsics[c_str]
            R_bc, t_bc = solve_pnp_for_pose(pattern_3d_custom, corners_2d, K, dist)
            if R_bc is not None:
                board2cam[c_str] = (R_bc, t_bc)
        if not board2cam:
            return False

        board_is_global= (pose_key==reference_pose)
        updated=False
        for c_str,(R_bc,t_bc) in board2cam.items():
            if c_str in camera_extrinsics:
                continue
            if board_is_global:
                camera_extrinsics[c_str]= (R_bc, t_bc)
                updated=True
            else:
                # bridging
                bridging_cam=None
                for c2 in board2cam.keys():
                    if c2!= c_str and c2 in camera_extrinsics:
                        bridging_cam=c2
                        break
                if bridging_cam is not None:
                    R_gC2, t_gC2= camera_extrinsics[bridging_cam]
                    R_bC2, t_bC2= board2cam[bridging_cam]
                    R_C2b, t_C2b= invert_rt(R_bC2, t_bC2)
                    R_gb, t_gb= compose_rt(R_gC2, t_gC2, R_C2b, t_C2b)
                    R_gc, t_gc= compose_rt(R_gb, t_gb, R_bc, t_bc)
                    camera_extrinsics[c_str]= (R_gc, t_gc)
                    updated=True
        return updated

    changed=True
    for _ in range(10):
        if not changed:
            break
        changed=False
        for p in valid_poses:
            if attempt_solve_extrinsics_for_pose(p):
                changed=True

    known_cameras= sorted(camera_extrinsics.keys())
    unknown_cameras= [c for c in cam_list if c not in known_cameras]
    if unknown_cameras:
        print("[WARN] Some cameras have no extrinsics =>", unknown_cameras)

    # 3) Print extrinsics
    print("\n=== Final Camera Extrinsic (Global->Camera) ===")
    for cam_str in known_cameras:
        R_gc, t_gc= camera_extrinsics[cam_str]
        print(f"Camera= {cam_str}")
        print("R=\n",R_gc)
        print("t=\n",t_gc.ravel())
        c_g= rt_to_cam_center(R_gc, t_gc).flatten()
        print("Center in global= ",c_g)

    # 4) Triangulate corners, EXACT as before
    def triangulate_point(obs_list):
        if len(obs_list)<2:
            return None
        (camA,(uA,vA))= obs_list[0]
        (camB,(uB,vB))= obs_list[1]
        if (camA not in camera_extrinsics or
            camA not in camera_intrinsics or
            camB not in camera_extrinsics or
            camB not in camera_intrinsics):
            return None
        R_gA, t_gA= camera_extrinsics[camA]
        (K_A, distA)= camera_intrinsics[camA]
        R_gB, t_gB= camera_extrinsics[camB]
        (K_B, distB)= camera_intrinsics[camB]

        P_A= K_A@ np.hstack([R_gA, t_gA])
        P_B= K_B@ np.hstack([R_gB, t_gB])

        ptA= np.array([uA,vA,1.0]).reshape(3,1)
        ptB= np.array([uB,vB,1.0]).reshape(3,1)

        X_hom= cv2.triangulatePoints(P_A, P_B, ptA[:2], ptB[:2])
        X_hom= X_hom.flatten()
        if abs(X_hom[3])< 1e-12:
            return None
        return X_hom[:3]/ X_hom[3]

    from collections import defaultdict
    triangulated_points= defaultdict(list)
    for p in pose_list:
        c_list= pose2cams[p]
        corner_map= defaultdict(list)
        for c_str in c_list:
            if (p, c_str) not in corners_dict:
                continue
            c2d= corners_dict[(p, c_str)]
            for i_pt in range(c2d.shape[0]):
                corner_map[i_pt].append((c_str, (c2d[i_pt,0], c2d[i_pt,1])))

        for idx,obs_list in corner_map.items():
            if len(obs_list)<2:
                continue
            X_3d= triangulate_point(obs_list)
            if X_3d is not None:
                triangulated_points[(p, idx)].append(X_3d)

    final_points_3d= {}
    for (p, idx),Xarr in triangulated_points.items():
        if len(Xarr)>0:
            final_points_3d[(p, idx)]= np.mean(Xarr, axis=0)

    print("[INFO] Triangulated", len(final_points_3d)," corners across all poses.")

    # 5) Plotly figure
    fig= go.Figure()

    def add_axis_line(fig, start, end, color, name, showlegend=False):
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines',
            line=dict(color=color, width=5),
            name=name,
            showlegend=showlegend
        ))

    axis_len= SQUARE_SIZE_MM*2
    O= np.array([0,0,0], dtype=float)
    add_axis_line(fig, O, O+[axis_len,0,0], 'red','GlobalX')
    add_axis_line(fig, O, O+[0,axis_len,0], 'green','GlobalY')
    add_axis_line(fig, O, O+[0,0,axis_len], 'blue','GlobalZ')

    # 5a) Plot triangulated corners EXACT as your code => color-coded
    import itertools
    color_cycle = itertools.cycle([
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
        "#ad494a", "#8c6d31", "#843c39", "#7b4173", "#5254a3", "#6b6ecf", "#637939", "#8ca252", "#bd9e39", "#e7ba52",
        "#e7969c", "#d6616b", "#cedb9c", "#393b79", "#3182bd", "#6baed6", "#9ecae1", "#b5cf6b", "#9c9ede", "#fbc15e"
    ])
    pose_color_map= {}
    for p in pose_list:
        pose_color_map[p]= next(color_cycle)
    from collections import defaultdict
    pose_xyz= defaultdict(lambda: ([],[],[]))
    for (p, idx), X_3d in final_points_3d.items():
        pose_xyz[p][0].append(X_3d[0])
        pose_xyz[p][1].append(X_3d[1])
        pose_xyz[p][2].append(X_3d[2])

    for p in pose_list:
        xs, ys, zs= pose_xyz[p]
        if len(xs)>0:
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='markers',
                marker=dict(size=4, color=pose_color_map[p]),
                name=f"Pose {p}"
            ))

    # 5b) Plot cameras => axes + centers + stl model
    cam_x, cam_y, cam_z= [],[],[]
    cam_labels= []
    for cam_str in known_cameras:
        R_gc, t_gc= camera_extrinsics[cam_str]
        center_g= rt_to_cam_center(R_gc, t_gc).flatten()

        cam_x.append(center_g[0])
        cam_y.append(center_g[1])
        cam_z.append(center_g[2])
        cam_labels.append(cam_str)

        # draw local axes => EXACT code
        draw_camera_axes(fig, center_g, R_gc, scale=100.0, cam_name=f"{cam_str}_axes")

        # add stl camera => do NOT alter the existing code
        draw_stl_camera_model(
            fig,
            stl_path= STL_FILE,
            stl_center_local= STL_CENTER,
            scale_factor= CAMERA_SCALE,
            R_gc= R_gc,
            t_gc= t_gc,
            name= f"{cam_str}_STL"
        )

    # 5c) add camera centers as usual
    fig.add_trace(go.Scatter3d(
        x=cam_x, y=cam_y, z=cam_z,
        mode='markers+text',
        text=cam_labels,
        textposition='middle right',
        marker=dict(color='magenta', size=5, symbol='circle'),
        name="Camera Centers"
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data'
        ),
        title="Camera + Chessboard + STL Model"
    )

    out_html= "Calibration3DVis_withChessAndSTL.html"
    pio.write_html(fig, file=out_html, auto_open=True)
    print("[INFO] Wrote interactive plot to", out_html)

    # ----------------------------------------------------------------------
    # NEW: Save system parameters (intrinsics & extrinsics) for later usage
    # ----------------------------------------------------------------------
    out_system_npz = "SystemParameters.npz"
    save_system_parameters_npz(
        out_system_npz,
        camera_intrinsics,
        camera_extrinsics,
        reference_pose
    )

    out_system_csv = "SystemParameters.csv"
    save_system_parameters_csv(
        out_system_csv,
        camera_intrinsics,
        camera_extrinsics,
        reference_pose
    )

    print("[DONE] All system parameters saved for future 3D recon usage.")

if __name__=="__main__":
    main()

#! /usr/bin/env python3
""" 
HeLiPR_Handler.py
    - Handler for HeLiPR dataset management and processing.
"""

import numpy as np
from pathlib import Path
import re
from typing import List
import struct

# NOTE: open3d, matplotlib, and tqdm are only needed for the
# visualisation / batch-loading helpers. They are imported lazily
# inside the methods that use them so that lightweight consumers
# (e.g. ROS 2 nodes that just want load_poses + load_scan_file)
# don't pay the import cost or require those packages installed.


class HeLiPR_Handler:
    def __init__(self, dataset_path:str, verbose:bool=True):
        """
        This is a class to handle the HeLiPR dataset, including loading, processing, and managing point cloud data
        Args:
            dataset_path (str): Path to the HeLiPR dataset. e.g., "/path/to/HeLiPR/"
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        Note:
            The dataset is expected to have a specific structure as defined by the HeLiPR dataset documentation:
            /path/to/HeLiPR/
                ├── Sample/
                │   ├── LiDAR/
                │   |   ├── Aeva/
                |   |   |   ├──1689519044559668850.bin
                |   |   |   ├──....bin
                │   |   ├── Avia/
                |   |   |   ├──....bin
                │   |   ├── Ouster/
                |   |   |   ├──....bin
                │   |   ├── Velodyne/
                |   |   |   ├──....bin
                │   ├── Undistorted/
                │   |   ├── Aeva/
                |   |   |   ├──1689519044559668850.bin
                |   |   |   ├──....bin
                │   |   ├── Avia/
                |   |   |   ├──....bin
                │   |   ├── Ouster/
                |   |   |   ├──....bin
                │   |   ├── Velodyne/
                |   |   |   ├──....bin
                │   ├── LiDAR_GT/
                │   |   ├── Aeva_gt.txt
                │   |   ├── Avia_gt.txt
                │   |   ├── Ouster_gt.txt
                │   |   ├── Velodyne_gt.txt
                |   |   ├── global_Aeva_gt.txt
                |   |   ├── global_Avia_gt.txt
                |   |   ├── global_Ouster_gt.txt
                |   |   ├── global_Velodyne_gt.txt
                │   ├── stamp.csv
                ├── Roundabout01/
                |   ├── ...
                ├── Roundabout02/
                └── ...
        """
        self.dataset_path = Path(dataset_path)
        self.verbose = verbose
        # Define extrinsic calibration matrices if needed
        self.T_velodyne2ouster = np.array([
            [ 0.999752289753613,      -0.0215240513828855,     0.00566342162254459,   -0.958469101466382],
            [ 0.0215555750390671,     0.999752161633017,     -0.00556529377889104,    0.0449041087926452],
            [-0.00554223034012042,    0.00568599350836049,    0.999968476083461,     -0.262799643355476],
            [0, 0, 0, 1]
        ])

        self.T_avia2ouster = np.array([
            [ 0.999821044774776,      0.0159071210418969,    0.0102392346214582,    0.628232128507671],
            [-0.0160846635739714,     0.999717526758036,    0.0174971509254758,   -0.338773055024622],
            [-0.00995801301399956,   -0.0176587143630322,   0.999794482773264,    -0.229651932062428],
            [0, 0, 0, 1]
        ])
        
        self.T_aeva2ouster = np.array([
            [ 0.999566194824045,      0.0248733202636163,   -0.0157714965695182,    0.541864230595932],
            [-0.0249842995614032,     0.999664174831162,   -0.00687912309516293,   0.369867831347708],
            [ 0.0155950934721411,     0.00727017869078270,  0.999851957822456,    -0.0869161258617896],
            [0, 0, 0, 1]
        ])
        
    
    ## Load the HeLiPR dataset for a given sequence and LiDAR sensor
    def load_helipr(self, sequence:str, lidar_sensor:str, type:str="Undistorted") -> dict:
        """
        Load the HeLiPR dataset for a given sequence and LiDAR sensor.
        Args:
            sequence (str): The sequence name, e.g., "Sample", "Roundabout01", etc.
            lidar_sensor (str): The LiDAR sensor name, e.g., "Aeva", "Avia", "Ouster", "Velodyne".
            type (str, optional): The type of point clouds to load, e.g., "Undistorted", "Database", "Query". Defaults to "Undistorted".
        Returns:
            dict: A dictionary containing the poses, pose timestamps, point clouds and point cloud timestamps.
        TODO: Extend this function to return the IMU data as well. 
        """
        seq_path = self.dataset_path / sequence 
        if not seq_path.exists():
            raise FileNotFoundError(f"Sequence path {seq_path} does not exist.")
        ## Load poses
        poses, p_timestamps = self.load_poses(seq_path, lidar_sensor)
        ## Load point clouds
        point_clouds, pc_timestamps = self.load_point_clouds(seq_path, lidar_sensor, type=type)
        ## Associate point clouds with closest poses based on timestamps
        associated_poses = []
        associated_timestamps = []
        for pc_ts in pc_timestamps:
            closest_idx = self.get_closest_pose_index(pc_ts, p_timestamps)
            associated_poses.append(poses[closest_idx])
            associated_timestamps.append(p_timestamps[closest_idx])
        poses = associated_poses
        p_timestamps = associated_timestamps
        ## If verbose, print summary
        if self.verbose:
            print(f"Loaded sequence '{sequence}' with LiDAR sensor '{lidar_sensor}':")
            print(f"  Number of poses: {len(poses)}")
            print(f"  Number of point clouds: {len(point_clouds)}")
        return {
            "poses": poses,
            "pose_timestamps": p_timestamps,
            "point_clouds": point_clouds,
            "pc_timestamps": pc_timestamps
        }
    
    ## Load poses for a given sequence and LiDAR sensor
    def load_poses(self, seq_path: Path, lidar_sensor: str) -> tuple:
        """
        Load poses for a given sequence and LiDAR sensor. The format is t, x, y, z, qx, qy, qz, qw.
        Args:
            seq_path (Path): Path to the sequence directory.
            lidar_sensor (str): The LiDAR sensor name. 
                - "Aeva", "Avia", "Ouster", or "Velodyne"
        Returns:
            tuple: A tuple containing:
                - poses (list of np.ndarray): List of poses as 4x4 transformation matrices.
                - timestamps (list of float): List of timestamps corresponding to each pose.
        """
        gt_poses_file = seq_path / "LiDAR_GT" / f"global_{lidar_sensor}_gt.txt"
        if not gt_poses_file.exists():
            raise FileNotFoundError(f"Ground truth poses file {gt_poses_file} does not exist.")
        
        poses = []
        timestamps = []
        with open(gt_poses_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                timestamp = float(parts[0])
                timestamp /= 1e9
                tx, ty, tz = map(float, parts[1:4])
                qx, qy, qz, qw = map(float, parts[4:8])
                # Convert quaternion to rotation matrix
                R = self._quaternion_to_rotation_matrix(qx, qy, qz, qw)
                # Construct transformation matrix
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = [tx, ty, tz]
                poses.append(T)
                timestamps.append(timestamp)
        return poses, timestamps
    
    def read_bin_file_original(self, filename, typeLiDAR):
        points = []
        with open(filename, "rb") as file:
            while True:
                if typeLiDAR == "Velodyne":
                    data = file.read(22)
                    if len(data) < 22:
                        break
                    x, y, z, intensity = struct.unpack('ffff', data[:16])
                    ring = struct.unpack('H', data[16:18])
                    time = struct.unpack('f', data[18:22])
                elif typeLiDAR == "Ouster":
                    data = file.read(26)
                    if len(data) < 26:
                        break
                    x, y, z, intensity = struct.unpack('ffff', data[:16])
                    t = struct.unpack('I', data[16:20]) 
                    reflectivity, ring, ambient = struct.unpack('HHH', data[20:26])
                elif typeLiDAR == "Aeva" and int(filename.split("/")[-1].split('.')[0]) > 1691936557946849179: 
                    data = file.read(29)
                    if len(data) < 29:
                        break
                    x, y, z, reflectivity, velocity = struct.unpack('fffff', data[:20])
                    time_offset_ns = struct.unpack('I', data[20:24])
                    line_index = struct.unpack('B', data[24:25])
                    intensity = struct.unpack('f', data[25:29])
                elif typeLiDAR == "Aeva" and int(filename.split("/")[-1].split('.')[0]) <= 1691936557946849179: 
                    data = file.read(25)
                    if len(data) < 25:
                        break
                    x, y, z, reflectivity, velocity = struct.unpack('fffff', data[:20])
                    time_offset_ns = struct.unpack('I', data[20:24])
                    line_index = struct.unpack('B', data[24:25])
                elif typeLiDAR == "Avia":
                    data = file.read(19)
                    if len(data) < 19:
                        break
                    x, y, z = struct.unpack('fff', data[:12])
                    reflectivity, tag, line = struct.unpack('BBB', data[12:15])
                    offset_time = struct.unpack('f', data[15:19])
                else:
                    raise ValueError("Unsupported LiDAR type")
                points.append([x, y, z])
        return points
    
    def read_bin_file_undistorted(self, filename: str) -> np.ndarray:
        """
        Read an undistorted .bin file from HeLiPR dataset.
        Args:
            filename (str): Path to the .bin file.
        Returns:
            np.ndarray: Nx3 array of points (x, y, z).
            float: Timestamp in seconds.
        """
        points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
        timestamp = Path(filename)
        timestamp = float(timestamp.stem) / 1e9  # Convert from nanoseconds to seconds
        return points[:, :3], timestamp  # Return only x, y, z
    
    def load_point_clouds(self, seq_path: Path, lidar_sensor: str, type: str = "Undistorted") -> tuple:
        """
        Load point clouds for a given sequence and LiDAR sensor.
        Args:
            seq_path (Path): Path to the sequence directory.
            lidar_sensor (str): "Aeva", "Avia", "Ouster", or "Velodyne"
            type (str, optional): Type of point clouds to load ("Undistorted", "LiDAR", "Database", "Query"). 
            Defaults to "Undistorted".
        Returns:
            (point_clouds, timestamps)
                point_clouds: list[np.ndarray] where each is (N,3) float32
                timestamps:   list[float] (NaN when a per-scan timestamp cannot be inferred)
        """
        lidar_path = seq_path / type / lidar_sensor
        if not lidar_path.exists():
            raise FileNotFoundError(f"LiDAR path {lidar_path} does not exist.")
        import tqdm  # lazy: only needed for progress bars
        bin_files = sorted(lidar_path.glob("*.bin"))
        point_clouds = []
        timestamps = []
        for bin_file in tqdm.tqdm(bin_files, desc=f"Loading point clouds from {lidar_sensor}", unit="file"):
            # Load point cloud from .bin file
            if type == "LiDAR":
                points = self.read_bin_file_original(str(bin_file), lidar_sensor)
                points = np.asarray(points)
                point_clouds.append(points)
                timestamp_str = bin_file.stem
                try:
                    timestamp = float(timestamp_str) / 1e9  # Convert from nanoseconds to seconds
                    timestamps.append(timestamp)
                except ValueError:
                    timestamps.append(float('nan'))  # If conversion fails, append NaN
            else:
                points, timestamp = self.read_bin_file_undistorted(str(bin_file))
                points = np.asarray(points)
                point_clouds.append(points)
                timestamps.append(timestamp)
        return point_clouds, timestamps

    # ----------------------------
    # Partial scan loading helpers
    # ----------------------------

    def list_scan_files(self, seq_path: Path, lidar_sensor: str, type: str = "Undistorted") -> List[Path]:
        """
        List .bin files for a given sequence/sensor without loading them.
        """
        lidar_path = seq_path / type / lidar_sensor
        if not lidar_path.exists():
            raise FileNotFoundError(f"LiDAR path {lidar_path} does not exist.")
        return sorted(lidar_path.glob("*.bin"))

    @staticmethod
    def _bin_timestamps(bin_files: List[Path]) -> List[float]:
        ts = []
        for bf in bin_files:
            try:
                ts.append(float(bf.stem) / 1e9)
            except ValueError:
                ts.append(float("nan"))
        return ts

    def select_scan_indices(
        self,
        bin_files: List[Path],
        anchor_frame: int,
        anchor_ts: float | None,
        num_scans: int,
    ) -> List[int]:
        """
        Select a forward window of scan indices starting at a frame or closest timestamp.
        """
        if not bin_files:
            return []

        if anchor_ts is not None:
            bin_ts = self._bin_timestamps(bin_files)
            valid_indices = [i for i, t in enumerate(bin_ts) if not np.isnan(t)]
            if not valid_indices:
                raise RuntimeError("No valid timestamps in scan directory.")
            closest_idx = min(valid_indices, key=lambda i: abs(bin_ts[i] - anchor_ts))
        else:
            closest_idx = max(0, min(anchor_frame, len(bin_files) - 1))

        start_idx = closest_idx
        end_idx = min(len(bin_files), start_idx + max(1, num_scans))
        return list(range(start_idx, end_idx))

    def load_scan_file(self, bin_file: Path, lidar_sensor: str, type: str = "Undistorted") -> tuple:
        """
        Load a single scan from a .bin file.
        Returns (points, timestamp_seconds).
        """
        if type == "Undistorted" or type == "Database" or type == "Query":
            points, timestamp = self.read_bin_file_undistorted(str(bin_file))
            points = np.asarray(points)
            return points, timestamp

        points = self.read_bin_file_original(str(bin_file), lidar_sensor)
        points = np.asarray(points)
        try:
            timestamp = float(bin_file.stem) / 1e9
        except ValueError:
            timestamp = float("nan")
        return points, timestamp

    def load_scans_window(
        self,
        seq_path: Path,
        lidar_sensor: str,
        type: str = "Undistorted",
        anchor_frame: int = 0,
        anchor_ts: float | None = None,
        num_scans: int = 1,
        radius: float = 0.0,
    ) -> tuple:
        """
        Load a small window of scans around a frame or timestamp.

        Returns:
            scans: list of (scan_index, points, timestamp, pose)
            indices: list of selected indices
            bin_files: list of all bin files for reference
            poses: list of poses aligned with scans
        """
        bin_files = self.list_scan_files(seq_path, lidar_sensor, type=type)
        if not bin_files:
            raise FileNotFoundError(f"No .bin files found for {lidar_sensor} in {seq_path}.")

        indices = self.select_scan_indices(bin_files, anchor_frame, anchor_ts, num_scans)
        scans = []

        poses, pose_ts = self.load_poses(seq_path, lidar_sensor)

        for idx in indices:
            pts, ts = self.load_scan_file(bin_files[idx], lidar_sensor, type=type)
            if radius > 0:
                r = np.linalg.norm(pts, axis=1)
                pts = pts[r <= radius]
            if pts.size == 0:
                continue

            pose = None
            if not np.isnan(ts):
                pose_idx = self.get_closest_pose_index(ts, pose_ts)
                pose = poses[pose_idx]
            scans.append((idx, pts, ts, pose))

        scan_poses = [s[3] for s in scans]
        return scans, indices, bin_files, scan_poses

    def build_submap_from_window(
        self,
        scans: List[tuple],
    ) -> tuple:
        """
        Build a submap by transforming all scans into the first scan's pose frame.

        Args:
            scans: list of (scan_index, points, timestamp, pose)

        Returns:
            submap_points: (N,3) points in first-pose frame
            ref_pose: 4x4 pose used as origin (first scan)
            ref_index: scan index in the original sequence (first element)
        """
        if not scans:
            raise ValueError("No scans provided.")

        ref_pose = scans[0][3]
        if ref_pose is None:
            raise RuntimeError("Reference pose is missing for the first scan.")

        ref_inv = np.linalg.inv(ref_pose)
        submap_pts = []

        for _, pts, _, pose in scans:
            if pose is None:
                continue
            pts_h = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
            pts_world = (pose @ pts_h.T).T
            pts_local = (ref_inv @ pts_world.T).T[:, :3].astype(np.float32)
            if pts_local.size > 0:
                submap_pts.append(pts_local)

        if not submap_pts:
            raise RuntimeError("No points transformed into submap.")

        submap = np.vstack(submap_pts)
        return submap, ref_pose, scans[0][0]

    ## For a given pointcloud timestamp, return the closest pose index
    def get_closest_pose_index(self, pc_timestamp: float, pose_timestamps: List[float]) -> int:
        """
        For a given pointcloud timestamp, return the closest pose index.
        Args:
            pc_timestamp (float): Timestamp of the point cloud.
            pose_timestamps (list of float): List of pose timestamps.
        Returns:
            int: Index of the closest pose.
        """
        closest_index = min(
            range(len(pose_timestamps)),
            key=lambda i: (pose_timestamps[i] - pc_timestamp) ** 2,
        )
        return closest_index

    ## Plot a live player of each point cloud in the sequence and the trajectory
    def plot_live_player(self, poses:list, pose_timestamps:list, point_clouds:list, pc_timestamps:list):
        import matplotlib.pyplot as plt  # lazy: visualisation only
        """
        Plot a live player of each point cloud (2D top view) in the sequence and the trajectory in matplotlib with two subplots.
        Start playing and pause with the "space" key. Next with the "right arrow" key. Previous with the "left arrow" key.
        Args:
            poses (list of np.ndarray): List of poses as 4x4 transformation matrices.
            pose_timestamps (list of float): List of timestamps corresponding to each pose.
            point_clouds (list of np.ndarray): List of point clouds as Nx3 numpy arrays.
            pc_timestamps (list of float): List of timestamps corresponding to each point cloud.
        """
        fig, (ax_pc, ax_traj) = plt.subplots(1, 2, figsize=(12, 6))
        plt.ion()
        plt.show()

        traj_x = []
        traj_y = []

        index = 0
        paused = False

        def on_key(event):
            nonlocal index, paused
            if event.key == ' ':
                paused = not paused
            elif event.key == 'right':
                index = min(index + 1, len(point_clouds) - 1)
            elif event.key == 'left':
                index = max(index - 1, 0)

        fig.canvas.mpl_connect('key_press_event', on_key)

        while True:
            ax_pc.clear()
            ax_traj.clear()

            # Plot point cloud (top view)
            pc = point_clouds[index]
            ax_pc.scatter(pc[:, 0], pc[:, 1], s=0.5)
            ax_pc.set_title(f"Point Cloud at {pc_timestamps[index]:.3f}s")
            ax_pc.set_xlabel("X (m)")
            ax_pc.set_ylabel("Y (m)")
            ax_pc.axis('equal')

            # Plot trajectory
            pose = poses[index]
            traj_x.append(pose[0, 3])
            traj_y.append(pose[1, 3])
            ax_traj.plot(traj_x, traj_y, '-o')
            # Format trajectory plot
            ax_traj.set_title("Trajectory")
            ax_traj.set_xlabel("X (m)")
            ax_traj.set_ylabel("Y (m)")
            ax_traj.axis('equal')
            # Set limits
            all_x = np.array(traj_x)
            all_y = np.array(traj_y)
            margin = 5
            ax_traj.set_xlim(all_x.min() - margin, all_x.max() + margin)
            ax_traj.set_ylim(all_y.min() - margin, all_y.max() + margin)
            # Add grid
            ax_traj.grid(True)
            ## Set fixed limits for point cloud view
            ax_pc.set_xlim(-100, 100)
            ax_pc.set_ylim(-100, 100)
            ax_pc.grid(True)
        
            plt.pause(0.1)

            if not paused:
                index += 1
                if index >= len(point_clouds):
                    break
        plt.ioff()
        plt.show()
        
    ## Plot the trajectory in 2D
    def plot_trajectory_2d(self, poses:list):
        import matplotlib.pyplot as plt  # lazy: visualisation only
        """
        Plot the trajectory in 2D using matplotlib.
        Args:
            poses (list of np.ndarray): List of poses as 4x4 transformation matrices.
        """
        traj_x = [pose[0, 3] for pose in poses]
        traj_y = [pose[1, 3] for pose in poses]

        plt.figure(figsize=(8, 8))
        plt.plot(traj_x, traj_y, '-o')
        plt.title("2D Trajectory")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis('equal')
        plt.grid(True)
        plt.show()
    
    ## Plot the trajectory in 2D for all LiDAR sensors in a sequence
    def plot_trajectory_2d_all_sensors(self, sequence:str, sensors:List[str]=["Aeva", "Avia", "Ouster", "Velodyne"]):
        import matplotlib.pyplot as plt  # lazy: visualisation only
        """
        Plot the trajectory in 2D for all LiDAR sensors in a sequence using matplotlib.
        Args:
            sequence (str): The sequence name, e.g., "Sample", "Roundabout01", etc.
            sensors (list of str, optional): List of LiDAR sensor names. Defaults to ["Aeva", "Avia", "Ouster", "Velodyne"].
        """
        plt.figure(figsize=(10, 10))
        for sensor in sensors:
            data = self.load_poses(self.dataset_path / sequence, sensor)
            # if sensor == "Velodyne":
            #     # Transform Ouster poses to Velodyne frame
            #     poses = [self.T_velodyne2ouster @ pose for pose in data[0]]
            # elif sensor == "Avia":
            #     poses = [self.T_avia2ouster @ pose for pose in data[0]]
            # elif sensor == "Aeva":
            #     poses = [self.T_aeva2ouster @ pose for pose in data[0]]
            # else:
            poses = data[0]
            traj_x = [pose[0, 3] for pose in poses]
            traj_y = [pose[1, 3] for pose in poses]
            plt.plot(traj_x, traj_y, '-o', label=sensor)
        
        plt.title(f"2D Trajectory for Sequence '{sequence}'")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    ## Plot the accumulated point cloud for a given sequence and LiDAR sensor
    def plot_accumulated_point_cloud(self, poses, pose_timestamps, point_clouds, pcd_timestamps, live:bool=False):
        import open3d as o3d  # lazy: visualisation only
        import tqdm           # lazy: progress bar only
        """
        Plot the accumulated point cloud for a given sequence and LiDAR sensor using Open3D.
        Args:
            poses (list of np.ndarray): List of poses as 4x4 transformation matrices.
            point_clouds (list of np.ndarray): List of point clouds as Nx3 numpy arrays.
            live (bool, optional): Whether to show the accumulation live. Defaults to False.
        """
        accumulated_pc = o3d.geometry.PointCloud()
        if not live:
            for pose, pc in tqdm.tqdm(zip(poses, point_clouds), total=len(point_clouds), desc="Accumulating point clouds"):
                o3d_pc = o3d.geometry.PointCloud()
                o3d_pc.points = o3d.utility.Vector3dVector(pc)
                # Transform point cloud to global frame
                o3d_pc.transform(pose)
                # Voxel downsample to reduce number of points
                o3d_pc = o3d_pc.voxel_down_sample(voxel_size=0.5)
                accumulated_pc += o3d_pc
            # save accumulated point cloud to file
            print(f"Total accumulated points: {len(accumulated_pc.points)}")
            o3d.io.write_point_cloud("accumulated_point_cloud_ouster_bridge02.pcd", accumulated_pc)
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Accumulated Point Cloud", width=800, height=600)
            vis.get_render_option().point_size = 1.0
            vis.get_render_option().background_color = np.array([0, 0, 0])
            vis.add_geometry(accumulated_pc)
            vis.run()
            vis.destroy_window()
            return
        else:
            for pose, pc in zip(poses, point_clouds):
                o3d_pc = o3d.geometry.PointCloud()
                o3d_pc.points = o3d.utility.Vector3dVector(pc)
                # Transform point cloud to global frame
                o3d_pc.transform(pose)
                accumulated_pc += o3d_pc
                vis.add_geometry(accumulated_pc)
                vis.poll_events()
                vis.update_renderer()
            vis.run()
            vis.destroy_window()
        
    ## Utility function to convert quaternion to rotation matrix
    def _quaternion_to_rotation_matrix(self, qx:float, qy:float, qz:float, qw:float) -> np.ndarray:
        """
        Convert a quaternion to a rotation matrix.
        Args:
            qx, qy, qz, qw (float): Quaternion components.
        Returns:
            np.ndarray: 3x3 rotation matrix.
        """
        R = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])
        return R


# if __name__ == "__main__":
#     # Example usage
#     dataset_path = "/home/niksta/Documents/datasets/HeLiPR/"
#     helipr_handler = HeLiPR_Handler(dataset_path, verbose=True)
#     data = helipr_handler.load_helipr(sequence="Bridge02", lidar_sensor="Ouster")
#     # helipr_handler.plot_trajectory_2d_all_sensors(sequence="Sample")
#     # helipr_handler.plot_trajectory_2d(data["poses"])
#     # helipr_handler.plot_live_player(
#     #     poses=data["poses"],
#     #     pose_timestamps=data["pose_timestamps"],
#     #     point_clouds=data["point_clouds"],
#     #     pc_timestamps=data["pc_timestamps"]
#     # )
#     # helipr_handler.plot_accumulated_point_cloud(
#     #     poses=data["poses"],
#     #     pose_timestamps=data["pose_timestamps"],
#     #     point_clouds=data["point_clouds"],
#     #     pcd_timestamps=data["pc_timestamps"],
#     #     live=False
#     # )
    

# ---------------------------------------------------------------------------
#  SequenceSource adapter
# ---------------------------------------------------------------------------
#
# `SENSOR_MAP` lived in evaluate_inlier_helipr.py, which meant the overlap-GT
# builder and the evaluation each parsed `--pair` with their own copy.  It
# belongs with the loader that knows what a HeLiPR sensor is.

SENSOR_MAP = {
    "O": "Ouster",
    "Ouster": "Ouster",
    "V": "Velodyne",
    "Velodyne": "Velodyne",
    "Aeva": "Aeva",
    "Avia": "Avia",
}


def resolve_sensor(tag: str) -> str:
    """``'O'`` -> ``'Ouster'``.  Raises on an unknown tag."""
    sensor = SENSOR_MAP.get(tag.strip())
    if sensor is None:
        raise ValueError(
            f"unknown HeLiPR sensor {tag!r}; valid names: {sorted(set(SENSOR_MAP))}"
        )
    return sensor


def parse_pair(pair: str) -> tuple:
    """``'O-Aeva'`` -> ``('Ouster', 'Aeva')``, DB sensor first."""
    parts = pair.split("-")
    if len(parts) != 2:
        raise ValueError(f"--pair must look like 'DB-Q', e.g. 'O-Aeva'; got {pair!r}")
    return resolve_sensor(parts[0]), resolve_sensor(parts[1])


class HeLiPRSource:
    """:class:`~inlier.eval.datasets.base.SequenceSource` over ``HeLiPR_Handler``."""

    name = "helipr"

    def __init__(
        self,
        dataset_path,
        sequence: str,
        sensor: str,
        scan_type: str = "Undistorted",
        verbose: bool = True,
    ) -> None:
        from pathlib import Path as _Path

        self.dataset_path = _Path(dataset_path)
        self.sequence = sequence
        self.sensor = resolve_sensor(sensor)
        # The raw HeLiPR scans are motion-distorted; every evaluation script
        # reads from Undistorted/, so that stays the default.
        self.scan_type = scan_type
        self.verbose = verbose
        self._handler = HeLiPR_Handler(str(self.dataset_path), verbose=verbose)

    def load(self, **_):
        from inlier.eval.datasets.base import Sequence

        data = self._handler.load_helipr(self.sequence, self.sensor, type=self.scan_type)
        return Sequence.from_handler_dict(data, **self.describe())

    def describe(self):
        return {
            "dataset_type": self.name,
            "dataset_path": str(self.dataset_path),
            "sequence": self.sequence,
            "sensor": self.sensor,
            "scan_type": self.scan_type,
        }

    @property
    def tag(self) -> str:
        return f"{self.sequence}_{self.sensor}"

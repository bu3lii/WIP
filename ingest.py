import os
import numpy as np
import open3d as o3d
import laspy
import pye57
import argparse

def load_pointCloud(path: str, downsample: float = None) -> o3d.geometry.PointCloud:
    ext = os.path.splitext(path)[1].lower()
    print(f" Loading Point Cloud: {path}")

    if ext in [".ply",".pcd"]:
        pcd = o3d.io.read_point_cloud(path)

    elif ext in [".xyz",".xyzrgb",".pts"]:
        data = np.loadtxt(path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:,:3])

        if data.shape[1] >= 6:
            colors = data[:,3:6].astype(np.float64)
            if colors.max() > 1.0:
                colors /= 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

    elif ext in [".las",".laz"]:
        las = laspy.read(path)
        pts = np.vstack((las.x,las.y,las.z)).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        if hasattr(las, "red") and hasattr(las, "green") and hasattr(las,"blue"):
            colors = np.vstack((las.red, las.green, las.blue)).T.astype(np.float64)
            colors /= np.max(colors)
            pcd.colors = o3d.utility.Vector3dVector(colors)

    elif ext == ".e57":
        e57 = pye57.E57(path)
        scan = e57.read_scan(0)
        pts = np.vstack((scan["cartesianX"], scan["cartesianY"], scan["cartesianZ"])).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if all(k in scan for k in ["colorRed","colorGreen","colorBlue"]):
            colors = np.vstack(
                (scan["colorRed"], scan["colorGreen"], scan["colorBlue"])
            ).T / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
    
    else:
        raise ValueError(f"Unsupported format: {ext}")

    if downsample:
        print(f" Downsampling poitn cloud (voxel_size={downsample})...")
        pcd = pcd.voxel_down_sample(voxel_size=downsample)

    print(f" Loaded {len(pcd.points)} points.")
    return pcd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal point cloud loader")
    parser.add_argument("input", help="Path to the point cloud file")
    parser.add_argument("--downsample", type=float, default=None, help="Voxel size for downsampling")
    parser.add_argument("--show", action="store_true", help="Visualize point cloud")
    args = parser.parse_args()

    pcd = load_pointCloud(args.input, args.downsample)

    if args.show:
        o3d.visualization.draw_geometries([pcd])
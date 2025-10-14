import open3d as o3d

def preprocess_pointcloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.005,
    remove_outliers: bool = True,
    estimate_normals: bool = True
) -> o3d.geometry.PointCloud:
    """
    Preprocess a point cloud for AI-ready pipelines.

    Steps:
      1. Downsample using voxel grid
      2. Optionally remove statistical outliers
      3. Optionally estimate surface normals

    Args:
        pcd: Input Open3D point cloud
        voxel_size: Size of voxel grid for downsampling (in meters)
        remove_outliers: Whether to apply statistical outlier removal
        estimate_normals: Whether to estimate normals for each point

    Returns:
        Cleaned and optionally downsampled Open3D PointCloud
    """
    print(f"[INFO] Preprocessing point cloud ({len(pcd.points)} points)...")

    # Downsample for uniform density
    if voxel_size:
        print(f"[INFO] Downsampling (voxel size = {voxel_size})...")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Remove isolated noise points
    if remove_outliers:
        print("[INFO] Removing statistical outliers...")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"[INFO] Retained {len(pcd.points)} points after noise removal.")

    # Estimate normals for geometric features
    if estimate_normals:
        print("[INFO] Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        pcd.normalize_normals()

    print(f"[INFO] Preprocessing complete: {len(pcd.points)} points ready.")
    return pcd

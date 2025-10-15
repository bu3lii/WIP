import open3d as o3d

def preprocess_pointcloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.005,
    remove_outliers: bool = True,
    estimate_normals: bool = True
) -> o3d.geometry.PointCloud:
    print(f"Preprocessing point cloud ({len(pcd.points)} points)...")

    # Downsample for uniform density
    if voxel_size:
        print(f"[INFO] Downsampling (voxel size = {voxel_size})...")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Remove isolated noise points
    if remove_outliers:
        print("Removing statistical outliers...")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"Retained {len(pcd.points)} points after noise removal.")

    # Estimate normals for geometric features
    if estimate_normals:
        print("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
        )
        pcd.normalize_normals()

    print(f"Preprocessing complete: {len(pcd.points)} points ready.")
    return pcd

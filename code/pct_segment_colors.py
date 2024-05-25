import open3d as o3d

# Load the point cloud
pcd = o3d.io.read_point_cloud("./frame_000375.ply")

import numpy as np

# Extract unique colors
unique_colors = np.unique(np.asarray(pcd.colors), axis=0)


# Initialize a dictionary to hold the point clouds
point_clouds = {}

for color in unique_colors:
    # Create a mask for the current color
    mask = np.all(pcd.colors == color, axis=1)

    # Extract the points of the current color
    points = np.asarray(pcd.points)[mask]

    # Create a new point cloud for the current color
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(np.repeat(color.reshape(1, -1), len(points), axis=0))

    # Add the point cloud to the dictionary
    point_clouds[str(color)] = point_cloud


    # Save the point clouds
for color, point_cloud in point_clouds.items():
    o3d.io.write_point_cloud(f"point_cloud_{color}.ply", point_cloud)

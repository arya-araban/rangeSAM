import numpy as np
import open3d as o3d
from sklearn.cluster import MeanShift

def load_point_cloud(file_path='point_cloud.ply'):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points

def save_point_cloud_with_centroids(file_path, points, centroids):
    # Convert points to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # Assign a default color to the original points
    default_color = [1, 1, 1]  # white
    pcd.paint_uniform_color(default_color)
    # Create a new point cloud for centroids
    centroids_pcd = o3d.geometry.PointCloud()
    centroids_pcd.points = o3d.utility.Vector3dVector(centroids)
    # Color the centroids in red
    red_color = [1, 0, 0]
    centroids_pcd.paint_uniform_color(red_color)
    # Merge the original point cloud with the centroids
    combined_pcd = pcd + centroids_pcd
    # Save the combined point cloud to file
    o3d.io.write_point_cloud(file_path, combined_pcd)

def find_density_peaks_mean_shift(point_cloud, bandwidth=1.0):
    # Apply Mean Shift
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(point_cloud)
    # The cluster centers are the centroids of high-density regions
    centroids = ms.cluster_centers_
    return centroids

def main(file_path='point_cloud.ply', output_file_path=None, visualize=False, bandwidth=1.0):
    point_cloud = load_point_cloud(file_path)
    centroids = find_density_peaks_mean_shift(point_cloud, bandwidth=bandwidth)
    # Print centroids
    print(f"number of centroids: {len(centroids)}")
    print("Centroids of high-density regions:", centroids)
    if visualize and output_file_path:
        save_point_cloud_with_centroids(output_file_path, point_cloud, centroids)
        print(f"Centroids visualization saved to {output_file_path}")

if __name__ == "__main__":
    # Specify the parameters here
    file_path = '/home/Arya/Work/PointSAM/code/CARLA Scripts/recordings/1716903731/frame_000269_segments/point_cloud_110_190_160.ply'
    output_file_path = 'centroids_visualization.ply'
    visualize = True
    bandwidth = 1.0

    main(file_path, output_file_path, visualize, bandwidth)

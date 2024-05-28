import numpy as np
import open3d as o3d
from sklearn.neighbors import KernelDensity
from scipy.ndimage import gaussian_filter, maximum_filter, label, find_objects

def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    return points

def save_point_cloud_with_centroids(file_path, points, centroids):
    # Convert points to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Assign a default color to the original points
    default_color = [0, 0, 0]  # black
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


def find_density_peaks_kde(point_cloud, bandwidth=1.0, grid_size=100):
    # Apply Kernel Density Estimation
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(point_cloud)

    # Create a dense grid over the 3D space
    grid_x, grid_y, grid_z = np.meshgrid(
        np.linspace(point_cloud[:,0].min(), point_cloud[:,0].max(), grid_size),
        np.linspace(point_cloud[:,1].min(), point_cloud[:,1].max(), grid_size),
        np.linspace(point_cloud[:,2].min(), point_cloud[:,2].max(), grid_size)
    )

    grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    log_density = kde.score_samples(grid_coords).reshape(grid_x.shape)

    # Apply Gaussian filter to smooth the density
    smoothed_density = gaussian_filter(np.exp(log_density), sigma=1)

    # Find local maxima using Non-Maximum Suppression
    local_max = maximum_filter(smoothed_density, size=5) == smoothed_density
    labeled, num_objects = label(local_max)
    slices = find_objects(labeled)

    centroids = []
    for s in slices:
        x_center = (grid_x[s[0].start:s[0].stop, s[1].start:s[1].stop, s[2].start:s[2].stop].mean())
        y_center = (grid_y[s[0].start:s[0].stop, s[1].start:s[1].stop, s[2].start:s[2].stop].mean())
        z_center = (grid_z[s[0].start:s[0].stop, s[1].start:s[1].stop, s[2].start:s[2].stop].mean())
        centroids.append([x_center, y_center, z_center])


    return np.array(centroids)  # convert list to numpy array

def main(file_path, output_file_path=None, visualize=False, bandwidth=1.0, grid_size=100):
    point_cloud = load_point_cloud(file_path)

    centroids = find_density_peaks_kde(point_cloud, bandwidth=bandwidth, grid_size=grid_size)

    # Print centroids
    print(f"number of centroids: {len(centroids)}")
    print("Centroids of high-density regions:", centroids)

    if visualize and output_file_path:
        save_point_cloud_with_centroids(output_file_path, point_cloud, centroids)
        print(f"Centroids visualization saved to {output_file_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Find high-density centroids in a 3D point cloud.')
    parser.add_argument('file_path', type=str, help='Input point cloud file path.')
    parser.add_argument('--output', type=str, help='Output point cloud file path for visualization.', default=None)
    parser.add_argument('--visualize', action='store_true', help='Visualize centroids by creating a new point cloud.')
    parser.add_argument('--bandwidth', type=float, default=3.0, help='Bandwidth for KDE.') # Default used to be 1
    parser.add_argument('--grid_size', type=int, default=50, help='Grid size for KDE.') # Default used to be 100

    args = parser.parse_args()

    main(args.file_path, args.output, args.visualize, args.bandwidth, args.grid_size)

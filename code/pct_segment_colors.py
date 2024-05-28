import open3d as o3d
import numpy as np
import os

def separate_colors(input_folder):
    # Ensure the input folder exists
    if not os.path.isdir(input_folder):
        raise ValueError(f"Input folder {input_folder} does not exist.")

    # List all PLY files in the input folder
    ply_files = [f for f in os.listdir(input_folder) if f.endswith('.ply')]

    if not ply_files:
        print("No PLY files found in the input folder.")
        return

    for ply_file in ply_files:
        # Load the point cloud
        pcd = o3d.io.read_point_cloud(os.path.join(input_folder, ply_file))

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

            # Add the point cloud to the dictionary with the color tuple as the key
            point_clouds[tuple(color)] = point_cloud

        # Create a directory for the segmented point clouds
        segment_dir = os.path.join(input_folder, f"{os.path.splitext(ply_file)[0]}_segments")
        os.makedirs(segment_dir, exist_ok=True)

        # Save the point clouds
        for color, point_cloud in point_clouds.items():
            # Clean up the color tuple to use it in filenames
            color_str = '_'.join(map(str, (np.array(color) * 255).astype(int)))  # Convert to 0-255 range and use as filename part
            output_file = os.path.join(segment_dir, f"point_cloud_{color_str}.ply")
            o3d.io.write_point_cloud(output_file, point_cloud)

if __name__ == "__main__":
    # Example usage
    input_folder = "/home/Arya/Work/PointSAM/code/CARLA Scripts/recordings/1716730729"
    separate_colors(input_folder)

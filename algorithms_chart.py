from mayavi import mlab
import os
os.environ['ETS_TOOLKIT'] = 'qt'

# Load the PLY file
mesh = mlab.pipeline.open('/home/yzi/Desktop/frame0_person0 (1).ply')

# Render the mesh
mlab.pipeline.surface(mesh, colormap='jet')

# Add lighting and axes
mlab.axes()
mlab.light()

# Show the plot in the notebook
mlab.show()
# %%
import open3d as o3d

# Create a visualizer object and window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add geometries to the visualizer
mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
vis.add_geometry(mesh)

# Set camera position and view angle
view_control = vis.get_view_control()
view_control.set_front([-0.4257, -0.2123, -0.8789])
view_control.set_lookat([0, 0, 0])
view_control.set_up([0.1752, -0.9753, 0.1358])
view_control.set_zoom(0.5379)

# Save camera view to JSON file
vis.get_render_option().save_to_json('camera_view.json')
# %%
import open3d as o3d
import os

# Path to directory containing .ply files
directory = '/home/yzi/Downloads/aik_3d_vis_results_mesh/'

# List all .ply files in directory
ply_files = [f for f in os.listdir(directory) if f.endswith('.ply')]

# Create a visualizer object and window
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add each .ply file to the visualizer
for ply_file in ply_files:
    # Load .ply file
    mesh = o3d.io.read_triangle_mesh(os.path.join(directory, ply_file))

    # Add mesh to visualizer
    vis.add_geometry(mesh)

# Set camera view and show visualizer
vis.get_render_option().load_from_json('camera_view.json')
vis.run()
# %%
import pyvista as pv

# Load the first ply file
mesh1 = pv.read('/home/yzi/Downloads/aik_3d_vis_results_mesh/mmhuman3d/aik_3d_vis_results_mesh/frame0_person0.ply')

# Load the second ply file
mesh2 = pv.read('/home/yzi/Downloads/aik_3d_vis_results_mesh/mmhuman3d/aik_3d_vis_results_mesh/frame31_person0.ply')

# Create a PyVista multi-block dataset
blocks = pv.MultiBlock([mesh1, mesh2])

# Create a PyVista plotter
plotter = pv.Plotter()

# Add the multi-block dataset to the plotter
plotter.add_mesh(blocks, opacity=0.5)

# Add a legend for the different blocks
plotter.add_legend([('Block 1', 'w'), ('Block 2', 'g')])

# Show the plot
plotter.show()
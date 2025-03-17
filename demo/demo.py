
"""
    Demonstration on how to build a stent with the given package.
    Here you find the option dictionary with a description to each of the parameters.

    {
        "centerline_params": {
            "centerline_path": "demo/centerline.mrk.json", - Path to centerline. Can be vtp of .mrk.json (3DSlicer)
            "x_smoothing": 0.01, - Smoothing of the centerline curve.
            "dx_smoothing": 0.001, - Smoothing of derivative curve.
            "s_start": 1 - Starting position of the stent (depends on the direction of the centerline!).
        },
        "stent_params": {
            "geo_params": {
            "w": 0.17, - Wire distance (see src/py_geo_fd/stent_data.json for more models).
            "lw": 13, - Wire length
            "d_nom": 2.5, - Nominal/maximal diameter (can be increased, but not recommended).
            "Nw": 52, - Total number of wires.
            "wire_radius": 0.015 - Wire radius.
            },
            "numeric_params": {
            "wire_resolution": 20, - Angular number of segments used to approximate the circular wires.
            "n_segments": 200 - Number of segments used to mesh the wires in axial direction.
            }
        },
        "wall_params": {
            "wall_path": "demo/surface.stl", - Path to aneurysm stl (or any other format supported by meshio with triangles).
            "smoothing": 30, - Number of gaußian smoothing steps of the wall surface.
            "margin": 0.0, - Margin, in case you need to shrink the wall surface (e.g. for a stent in a stent).
            "n_axial_segments": 128, - Axial resolution of the wall distance computation.
            "n_radial_segments": 72 - Radial resolution of the wall distance computation.
        },
        "n_cpu": 14 - Number of CPUs used for the wall computation.
    }
"""

from py_geo_fd import Stent

# Build stent using the config file
stent = Stent("demo/stent_config.json")

# Save the centerline after smoothing. This helps bebugging code and checking that the smoothing is not too large/small.
stent.save_centerline("demo/centerline.vtp")

# Save the stent envelope including the local porosity and wire angles.
stent.save_envelope("demo/stent_envelope")

# Save the stent as a collection of 1D lines as vtp (ideal for embedding)
stent.save_stent_to_vtp("demo/stent.vtp")

# Save the stent as a tetrahaedra mesh (ideal for visualization). It will output sets of positive and negative winding wires for potential embedding.
# The outputs are in the vtu format. No need to specifying that in the path.
stent.save_stent("demo/stent")

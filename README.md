# Camera-Pose-Verifier

<center>
<image src="img.png">
</center>

Camera coordinate systems can make me cry, hopefully not anymore.

This is a modified version of Pytorch3D's [camera pose visualizer](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/vis/plotly_vis.py#L104) - meant to simplify the process of visualizing camera poses. It accepts a list of poses and displays it in a 3D grid while being flexible enough to handle different coordinate systems. It can be used as follows:

```python
from vis_pose import plot_cameras

# poses is a list of 4x4 numpy arrays
poses = [...]
fig = plot_cameras(poses, coordinate_frame='opengl', camera_scale=0.3, scheme='rainbow', c2w=True)
```

The arguments are:
- poses: A 4x4 numpy array of poses
- coordinate_frame: The coordinate frame of the poses. Can be 'opengl', 'opencv', 'blender', 'pytorch3d'. Internally, all camera poses are converted to opencv coordinate frame for visualisation.
- camera_scale: The scale of the camera in the visualisation
- scheme: The colour scheme of the cameras. Can be anything that matplotlib accepts as a colour scheme.
- c2w: Whether the poses are camera-to-world or world-to-camera. If True, the poses are assumed to be camera-to-world, and if False, the poses are assumed to be world-to-camera.

# Requirements
- pytorch3d
- plotly
- matplotlib
- torch

# To Do

[] Verify conversion from blender/Pytorch3D to opencv \
[] Make cameras opaque/not wireframes \
[] Remove Pytorch3D dependency

# Useful References

- https://ksimek.github.io/2012/08/22/extrinsic/ - What an extrinsic matrix is

- https://ai-workshops.github.io/building-and-working-in-environments-for-embodied-ai-cvpr-2022/Section-4-Debug.pdf - Information on various coordinate systems used in CV/Robotics

- https://github.com/demul/extrinsic2pyramid - A great tool for visualizing camera poses (but assumes opencv coordinate system + matplotlib). This repo is just a more flexible version of this. Further, since it uses plotly, it is interactive and one can even disable certain poses from the visualisation as required.

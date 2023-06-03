import warnings
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import plotly.graph_objects as go
import torch
import glob
import os
import numpy as np
from plotly.subplots import make_subplots
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.utils import cameras_from_opencv_projection

import matplotlib.cm as cm

class AxisArgs(NamedTuple):  # pragma: no cover
    showgrid: bool = False
    zeroline: bool = False
    showline: bool = False
    ticks: str = ""
    showticklabels: bool = False
    backgroundcolor: str = "#fff"
    showaxeslabels: bool = False

def convert_poses_opencv(poses: List[np.array], coordinate_frame: str) -> List[np.array]:

    if coordinate_frame == 'opengl':
        transform_to_opencv = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ])
    elif coordinate_frame == 'blender':
        transform_to_opencv = np.array([
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
        ])
    elif coordinate_frame == 'pytorch3d':
        transform_to_opencv = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ])
    else:
        raise ValueError(f'Unknown coordinate frame {coordinate_frame}')
    
    poses_opencv = []
    for pose in poses:
        pose_opencv = transform_to_opencv @ pose
        poses_opencv.append(pose_opencv)
    
    return poses_opencv

def get_camera_wireframe(scale: float = 0.3):  # pragma: no cover
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    a = 0.5 * torch.tensor([-2, 1.5, 4])
    up1 = 0.5 * torch.tensor([0, 1.5, 4])
    up2 = 0.5 * torch.tensor([0, 2, 4])
    b = 0.5 * torch.tensor([2, 1.5, 4])
    c = 0.5 * torch.tensor([-2, -1.5, 4])
    d = 0.5 * torch.tensor([2, -1.5, 4])
    C = torch.zeros(3)
    F = torch.tensor([0, 0, 3])
    camera_points = [a, up1, up2, up1, b, d, c, a, C, b, d, C, c, C, F]
    lines = torch.stack([x.float() for x in camera_points]) * scale
    return lines


def _gen_fig_with_subplots(
    batch_size: int, ncols: int, subplot_titles: List[str]
):  # pragma: no cover
    """
    Takes in the number of objects to be plotted and generate a plotly figure
    with the appropriate number and orientation of titled subplots.
    Args:
        batch_size: the number of elements in the batch of objects to be visualized.
        ncols: number of subplots in the same row.
        subplot_titles: titles for the subplot(s). list of strings of length batch_size.

    Returns:
        Plotly figure with ncols subplots per row, and batch_size subplots.
    """
    fig_rows = batch_size // ncols
    if batch_size % ncols != 0:
        fig_rows += 1  # allow for non-uniform rows
    fig_cols = ncols
    fig_type = [{"type": "scene"}]
    specs = [fig_type * fig_cols] * fig_rows
    # subplot_titles must have one title per subplot
    fig = make_subplots(
        rows=fig_rows,
        cols=fig_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        column_widths=[1.0] * fig_cols,
    )
    return fig


def _update_axes_bounds(
    verts_center: torch.Tensor,
    max_expand: float,
    current_layout: go.Scene,  # pyre-ignore[11]
) -> None:  # pragma: no cover
    """
    Takes in the vertices' center point and max spread, and the current plotly figure
    layout and updates the layout to have bounds that include all traces for that subplot.
    Args:
        verts_center: tensor of size (3) corresponding to a trace's vertices' center point.
        max_expand: the maximum spread in any dimension of the trace's vertices.
        current_layout: the plotly figure layout scene corresponding to the referenced trace.
    """
    verts_center = verts_center.detach().cpu()
    verts_min = verts_center - max_expand
    verts_max = verts_center + max_expand
    bounds = torch.t(torch.stack((verts_min, verts_max)))

    # Ensure that within a subplot, the bounds capture all traces
    old_xrange, old_yrange, old_zrange = (
        current_layout["xaxis"]["range"],
        current_layout["yaxis"]["range"],
        current_layout["zaxis"]["range"],
    )
    x_range, y_range, z_range = bounds
    if old_xrange is not None:
        x_range[0] = min(x_range[0], old_xrange[0])
        x_range[1] = max(x_range[1], old_xrange[1])
    if old_yrange is not None:
        y_range[0] = min(y_range[0], old_yrange[0])
        y_range[1] = max(y_range[1], old_yrange[1])
    if old_zrange is not None:
        z_range[0] = min(z_range[0], old_zrange[0])
        z_range[1] = max(z_range[1], old_zrange[1])

    xaxis = {"range": x_range}
    yaxis = {"range": y_range}
    zaxis = {"range": z_range}
    current_layout.update({"xaxis": xaxis, "yaxis": yaxis, "zaxis": zaxis})

def _add_camera_trace(
    fig: go.Figure,
    cameras: CamerasBase,
    trace_name: str,
    subplot_idx: int,
    ncols: int,
    camera_scale: float,
    color: List[float],
    opacity: float = 1.0,
) -> None:  # pragma: no cover
    """
    Adds a trace rendering a Cameras object to the passed in figure, with
    a given name and in a specific subplot.

    Args:
        fig: plotly figure to add the trace within.
        cameras: the Cameras object to render. It can be batched.
        trace_name: name to label the trace with.
        subplot_idx: identifies the subplot, with 0 being the top left.
        ncols: the number of subplots per row.
        camera_scale: the size of the wireframe used to render the Cameras object.
    """
    cam_wires = get_camera_wireframe(camera_scale).to(cameras.device)
    cam_trans = cameras.get_world_to_view_transform().inverse()
    cam_wires_trans = cam_trans.transform_points(cam_wires).detach().cpu()
    # if batch size is 1, unsqueeze to add dimension
    if len(cam_wires_trans.shape) < 3:
        cam_wires_trans = cam_wires_trans.unsqueeze(0)

    nan_tensor = torch.Tensor([[float("NaN")] * 3])
    all_cam_wires = cam_wires_trans[0]
    for wire in cam_wires_trans[1:]:
        # We combine camera points into a single tensor to plot them in a
        # single trace. The NaNs are inserted between sets of camera
        # points so that the lines drawn by Plotly are not drawn between
        # points that belong to different cameras.
        all_cam_wires = torch.cat((all_cam_wires, nan_tensor, wire))
    x, y, z = all_cam_wires.detach().cpu().numpy().T.astype(float)

    # Scale colours to 255 int
    color = [int(c * 255) for c in color]
    color[-1] = opacity

    row, col = subplot_idx // ncols + 1, subplot_idx % ncols + 1
    fig.add_trace(
        go.Scatter3d(x=x, y=y, z=z, marker={
            "size": 1,
            "color": f"rgba{str(tuple(color))}",
        }, name=trace_name),
        row=row,
        col=col,
    )

    # Access the current subplot's scene configuration
    plot_scene = "scene" + str(subplot_idx + 1)
    current_layout = fig["layout"][plot_scene]

    # flatten for bounds calculations
    flattened_wires = cam_wires_trans.flatten(0, 1)
    verts_center = flattened_wires.mean(0)
    max_expand = (flattened_wires.max(0)[0] - flattened_wires.min(0)[0]).max()
    _update_axes_bounds(verts_center, max_expand, current_layout)

def plot_cameras(poses, coordinate_frame='opengl', camera_scale=0.3, scheme='rainbow', c2w=True, **kwargs):

    # Convert poses back to opengl if not in opengl format
    if coordinate_frame != 'opencv':
        poses = convert_poses_opencv(poses, coordinate_frame)

    if c2w:
        poses = [np.linalg.inv(pose) for pose in poses] # Pytorch3D uses world-to-camera convention
    
    K = torch.eye(3)[None, ...]
    K[..., 0, 0] = K[..., 1, 1] = 512
    K[..., 0, 2] = K[..., 1, 2] = 256

    poses = [cameras_from_opencv_projection(
        torch.from_numpy(pose[:3, :3][None, ...]),
        torch.from_numpy(pose[:3, 3][None, ...]),
        K,
        torch.ones(1, 2) * 512,
    ) for pose in poses]

    fig = _gen_fig_with_subplots(1, 1, ["All Poses"])
    axis_args_dict = kwargs.get("axis_args", AxisArgs())._asdict()

    # Set axis arguments to defaults defined at the top of this file
    x_settings = {**axis_args_dict}
    y_settings = {**axis_args_dict}
    z_settings = {**axis_args_dict}

    # Update the axes with any axis settings passed in as kwargs.
    x_settings.update(**kwargs.get("xaxis", {}))
    y_settings.update(**kwargs.get("yaxis", {}))
    z_settings.update(**kwargs.get("zaxis", {}))

    camera = {
        "up": {
            "x": 0.0,
            "y": 1.0,
            "z": 0.0,
        }  # set the up vector to match PyTorch3D world coordinates conventions
    }
    
    colormap = cm.get_cmap(scheme)
    colors = colormap(np.linspace(0, 1, len(poses)))

    for i, pose in enumerate(poses):
        _add_camera_trace(fig, pose, i, 0, 1, camera_scale, colors[i])

    return fig

if __name__ == '__main__':

    # Test the camera pose visualization
    poses_dir = "test_poses"

    pose_files = sorted(glob.glob(os.path.join(poses_dir, "2_*.txt")))
    poses_cv  = [np.loadtxt(pose_file) for pose_file in pose_files]

    # Plot
    fig = plot_cameras(poses_cv, coordinate_frame='opencv', scheme='rainbow', c2w=True)
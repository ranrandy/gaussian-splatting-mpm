import numpy as np
import tensorflow as tf

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def normalize(v):
    return tf.math.l2_normalize(v, axis=-1, epsilon=1e-6)

def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    hwf = c2w[:,4:5]
    rads = np.array(list(rads) + [1.])
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.sin(-theta), np.cos(-theta), np.sin(-theta * zrate), 1.]) * rads)
        z = normalize(-c + np.dot(c2w[:3,:4], np.array([0, 0, focal, 1.])))
        pose = np.eye(4)
        pose[:3, :4] = viewmatrix(z, up, c)
        render_poses.append(pose)

    return render_poses

def get_render_poses_spiral(focal_length, bounds_data, intrinsics, poses, args, N_views=60, N_rots=2):
    intrinsics = np.array(intrinsics)
    poses = np.array(poses)

    ## Focus distance
    if focal_length < 0:
        close_depth, inf_depth = bounds_data.min() * .9, bounds_data.max() * 5.
        dt = .75
        mean_dz = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)
        focal_length = mean_dz

    # Get average pose
    c2w = poses_avg(poses)
    c2w_path = c2w
    up = normalize(poses[:, :3, 1].sum(0))

    # Get radii for spiral path
    shrink_factor = .8
    zdelta = bounds_data.min() * .2
    tt = (poses[:, :3, 3] - c2w[:3, 3])
    if np.sum(tt) == 0.0:
        tt = np.array([1.0, 1.0, 1.0])

    rads = np.percentile(np.abs(tt), 90, 0) \
        * np.array([args.rad_multiplier_x, args.rad_multiplier_y, args.rad_multiplier_z])
    light_rads = np.percentile(np.abs(tt), 90, 0) \
        * np.array([args.rad_multiplier_x, args.rad_multiplier_y, args.rad_multiplier_z])

    # Generate poses for spiral path
    render_poses = render_path_spiral(c2w_path, up, rads, focal_length, zdelta, zrate=.5, rots=N_rots, N=N_views)
    render_poses = np.array(render_poses).astype(np.float32)

    render_light_poses = render_path_spiral(c2w_path, up, light_rads, focal_length, zdelta, zrate=.5, rots=N_rots, N=N_views)
    render_light_poses = np.array(render_light_poses).astype(np.float32)

    return render_poses, render_light_poses
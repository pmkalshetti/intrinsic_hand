import numpy as np


def create_4x4_trans_mat_from_R_t(R=np.eye(3), t=np.zeros(3)):
    mat_view = np.array([
        [R[0, 0], R[0, 1], R[0, 2], t[0]],
        [R[1, 0], R[1, 1], R[1, 2], t[1]],
        [R[2, 0], R[2, 1], R[2, 2], t[2]],
        [      0,       0,       0,    1],
    ])
    return mat_view

def create_proj_mat_from_pinhole_camera(width, height, fx, fy, cx, cy, near, far):
    # Ref: https://stackoverflow.com/questions/22064084/how-to-create-perspective-projection-matrix-given-focal-points-and-camera-princ/57335955
    # Ref: http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/

    # X = (u - cu) * Z / fx # cu and cx both are same, principal point
    left = -cx * near / fx  # set u = 0 in above formula, left of image plane has u = 0
    top = cy * near / fy    # top of image plane has v = 0
    right = (width - cx) * near/fx    # right of image plane has u = width
    bottom = -(height - cy) * near/fy   # bottom of image plane has v = height
    mat_projection = np.array([
        [2*near/(right-left),                   0, (right+left)/(right-left),                      0],
        [                  0, 2*near/(top-bottom), (top+bottom)/(top-bottom),                      0],
        [                  0,                   0,    -(far+near)/(far-near), -2*far*near/(far-near)],
        [                  0,                   0,                        -1,                      0]
    ])

    return mat_projection

def create_proj_mat(left, right, bottom, top, near, far):
    return np.array([
        [2*near/(right-left),                   0, (right+left)/(right-left),                      0],
        [                  0, 2*near/(top-bottom), (top+bottom)/(top-bottom),                      0],
        [                  0,                   0,    -(far+near)/(far-near), -2*far*near/(far-near)],
        [                  0,                   0,                        -1,                      0]
    ])

def create_ortho_proj_mat_from_pinhole_camera(width, height, fx, fy, cx, cy, near, far):
    # Ref: https://stackoverflow.com/questions/22064084/how-to-create-perspective-projection-matrix-given-focal-points-and-camera-princ/57335955
    
    # X = (u - cu) * Z / fx # cu and cx both are same, principal point
    left = -cx * near / fx  # set u = 0 in above formula, left of image plane has u = 0
    top = cy * near / fy    # top of image plane has v = 0
    right = (width - cx) * near/fx    # right of image plane has u = width
    bottom = -(height - cy) * near/fy   # bottom of image plane has v = height
    l = left; t = top; r = right; b = bottom; f = far; n = near
    mat_projection = np.array([
        [2/(r-l),       0,       0,  -(r+l)/(r-l)],
        [      0, 2/(t-b),       0,  -(t+b)/(t-b)],
        [      0,       0, 2/(f-n),  -(f+n)/(f-n)],
        [      0,       0,       0,             1]
    ])

    return mat_projection



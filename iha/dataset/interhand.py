import numpy as np
import torch
import json
import cv2 as cv
from pathlib import Path
from pycocotools.coco import COCO
import shutil
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import subprocess
import plotly.graph_objects as go

from ..mano.mano import Mano
from .. import utils

logger = utils.get_logger(__name__)

def cnt_area(cnt):
    area = cv.contourArea(cnt)
    return area

def obtain_mask_from_verts_img(img_raw, verts_img, faces):
    # create convex polygon from vertices in image space
    # Ref: https://github.com/SeanChenxy/HandAvatar/blob/3b1c70b9d8d829bfcea1255743daea6dd8ed0b1d/segment/seg_interhand2.6m_from_mano.py#L210
    mask = np.zeros_like(img_raw)
    for f in faces:
        triangle = np.array([
            [verts_img[f[0]][0], verts_img[f[0]][1]],
            [verts_img[f[1]][0], verts_img[f[1]][1]],
            [verts_img[f[2]][0], verts_img[f[2]][1]],
        ])
        cv.fillConvexPoly(mask, triangle, (255, 255, 255))
    
    # filter mask
    if mask.max()<20:
        print(f"mask is all black")
        return None
    mask_bool = mask[..., 0]==255
    sel_img = img_raw[mask_bool].mean(axis=-1)
    if sel_img.max()<20:
        print(f"sel_img is all black")
        return None
    sel_img = np.bitwise_and(sel_img>10, sel_img<200)
    mask_bool[mask_bool] = sel_img.astype('int32')
    mask = mask * mask_bool[..., None]
    contours, _ = cv.findContours(mask[..., 0], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = list(contours)
    contours.sort(key=cnt_area, reverse=True)
    poly = contours[0].transpose(1, 0, 2).astype(np.int32)
    poly_mask = np.zeros_like(mask)
    poly_mask = cv.fillPoly(poly_mask, poly, (1,1,1))
    mask = mask * poly_mask

    return mask

def preprocess_interhand(cfg, interhand_root_dir, framelist, split, capture_id_selected, seq_name_selected, cam_id_selected, frame_id_selected, out_preprocess_dir):
    """
    preprocessing steps:
        1. segment image using mano annotations
        2. split data into training and validation
    """
    def world2cam(world_coord, R, T):
        cam_coord = np.dot(R, world_coord - T)
        return cam_coord
    
    def cam2pixel(cam_coord, f, c):
        x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
        y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
        z = cam_coord[:, 2]
        img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
        return img_coord

    # ==============================================================================================
    # output directories
    # ==============================================================================================
    out_dir = out_preprocess_dir; utils.create_dir(out_dir, True)
    out_img_raw_dir = f"{out_dir}/img_raw"; utils.create_dir(out_img_raw_dir, True)
    out_img_seg_dir = f"{out_dir}/img_seg"; utils.create_dir(out_img_seg_dir, True)
    out_mano_param_dir = f"{out_dir}/mano_param"; utils.create_dir(out_mano_param_dir, True)
    out_mano_verts_dir = f"{out_dir}/mano_verts"; utils.create_dir(out_mano_verts_dir, True)
    out_mv_dir = f"{out_dir}/mv"; utils.create_dir(out_mv_dir, True)
    out_mvp_dir = f"{out_dir}/mvp"; utils.create_dir(out_mvp_dir, True)
    
    mano = Mano(cfg.MANO)

    # ==============================================================================================
    #  read annotations
    # ==============================================================================================
    db = COCO(f"{interhand_root_dir}/annotations/{split}/InterHand2.6M_{split}_data.json")
    joint_num = 21 # single hand
    joint_type = {"right": np.arange(0, joint_num), "left": np.arange(joint_num, joint_num*2)}
    with open(f"{interhand_root_dir}/annotations/{split}/InterHand2.6M_{split}_camera.json") as f:
        cameras = json.load(f)
    with open(f"{interhand_root_dir}/annotations/{split}/InterHand2.6M_{split}_joint_3d.json") as f:
        joints = json.load(f)
    with open(f"{interhand_root_dir}/annotations/{split}/InterHand2.6M_{split}_MANO_NeuralAnnot.json") as f:
        mano_params = json.load(f)

    if framelist:
        with open(framelist, "r") as file:
            # list_cam = []
            # list_frame = []
            list_filename = []
            # list_id_frame = [line for line in file]
            for line in file:
                # _split, _cap, _seq, _cam, _frame = line.split("/")
                # _cam = _cam[3:] # remove cam prefix
                # _frame = _frame[5:-5]   # remove image prefix and .jpg\n suffix
                # list_cam.append(_cam)
                # list_frame.append(_frame)
                _filename = line.removeprefix("test/")[:-1] # [:-1] removes \n 
                list_filename.append(_filename)

    i_data = 0
    for id_ann, aid in enumerate(tqdm(db.anns.keys())):
        ann = db.anns[aid]
        hand_type = ann['hand_type']
        
        image_id = ann['image_id']
        img_data = db.loadImgs(image_id)[0]
        capture_id = img_data['capture']
        seq_name = img_data['seq_name']
        cam = img_data['camera']
        frame_idx = img_data['frame_idx']
        filename = img_data['file_name']
        if framelist:
            # in_framelist = False
            # for item_cam, item_frame in zip(list_cam, list_frame):
            #     if (cam == item_cam) and (str(frame_idx) == item_frame):
            #         in_framelist = True
            # if not in_framelist:
            #     continue
            if not (filename in list_filename):
                continue
            
        else:
            if hand_type != "right":
                continue
            if capture_id != capture_id_selected:
                continue
            if seq_name != seq_name_selected:
                continue
            if cam_id_selected != "all" and cam != cam_id_selected:
                continue
            if frame_id_selected != "all" and str(frame_idx) != frame_id_selected:
                continue
 
        try:
            mano_param = mano_params[str(capture_id)][str(frame_idx)][hand_type]
            # mano_param["shape"].shape: (10,)
            # mano_param["pose"].shape: (48,)  mean pose subtracted from hand pose (i.e. from the last 45 elements)
            # mano_param["trans"].shape: (3,)
            if mano_param is None:
                continue
        except KeyError:
            logger.warning(f"{id_ann}, cannot read mano params, {filename}")
            continue

        img_width, img_height = img_data['width'], img_data['height']
        bbox = np.array(ann['bbox'], dtype=np.float32) # x,y,w,h
        if not framelist:
            if bbox[0]<10 or bbox[1]<10 or max(bbox[2], bbox[3])<80 or bbox[0]+bbox[2]>img_width-10 or bbox[1]+bbox[3]>img_height-10:
                logger.warning(f"{id_ann}, Discard {filename}, bbox is too biased/small: {bbox.tolist()}")
                continue
        
        img_path = f"{interhand_root_dir}/images/{split}/{filename}"
        img_raw = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)  # (512, 334, 3)
        if not framelist:
            if img_raw.max() < 20:
                logger.warning(f"{id_ann}, Discard {filename}, RGB is too dark: {img_raw.max()}")
                continue
            if np.allclose(img_raw[..., 0], img_raw[..., 1], atol=1) or np.allclose(img_raw[..., 2], img_raw[..., 1], atol=1) or np.allclose(img_raw[..., 0], img_raw[..., 2], atol=1):
                logger.warning(f"{id_ann}, Discard {filename}, Gray scale")
                continue
        

        # parse camera parameters
        campos, camrot = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float32)
        # (3,), (3, 3)
        focal, princpt = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float32), np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float32)
        # (2,), (2,)

        pose = np.array(mano_param["pose"])
        shape = np.array(mano_param["shape"])
        trans = np.array(mano_param["trans"])
        # get MANO 3D mesh coordinates (world coordinate)
        mano_out = mano(betas=torch.from_numpy(shape)[None, :].float(), global_orient=torch.from_numpy(pose)[None, :3].float(), hand_pose=torch.from_numpy(pose)[None, 3:].float(), transl=torch.from_numpy(trans)[None, :].float())
        verts_world = mano_out.vertices[0].numpy()   # in m

        # apply camera extrinsics
        # Ref: building extrinsic matrix from camera pose at https://ksimek.github.io/2012/08/22/extrinsic/
        # consider R, t be the orientation and position of the camera in world coordinates (i.e., it denotes the camera pose)
        # The extrinsic matrix (denotes how points in world coordinates should transform to camera coordinates) is then given by 
        # [
        #   [R.T, -R.T@t],
        #   [0, 1]
        # ]
        # Ref: https://github.com/facebookresearch/InterHand2.6M/blob/67ba1b8e2c8da0f79ba8a3de5bb401714b4ebea2/tool/MANO_render/render.py#L124
        R = camrot # Note: .T is not required, why? (possibly because cam_rot is already transposed)
        t = - R @ campos / 1000 # campos mm to m
        verts_cam = (R @ verts_world.T).T + t

        # apply intrinsic
        K = np.array([
            [focal[0],        0, princpt[0]],
            [       0, focal[1], princpt[1]],
            [       0,        0,          1]
        ], dtype=np.float32)
        verts_img = (K @ verts_cam.T).T
        verts_img[:, :2] = np.round(verts_img[:, :2] / verts_img[:, 2:3])
        verts_img = verts_img.astype(np.int32)
        # obtain mask
        mask = obtain_mask_from_verts_img(img_raw, verts_img, mano.faces)
        if not framelist:
            if mask is None:
                # print(f'{i}, Discard {image_name}, w/o mask')
                logger.warning(f"{id_ann}, Discard {filename}, w/o mask")
                continue
            mask_sum = mask[..., 0].astype('bool').sum()
            if mask.max() < 255 or mask_sum < 3000:
                logger.warning(f"{id_ann}, Discard {filename}, mask is too dark: {mask.max()}, {mask_sum}")
                continue
        
        img_seg = np.zeros_like(img_raw)
        m_valid, n_valid = np.nonzero(mask[:, :, 0])
        img_seg[m_valid, n_valid, :] = img_raw[m_valid, n_valid, :]
        
        # crop and resize image
        x, y, w, h = bbox
        x1 = np.max((0, x))
        y1 = np.max((0, y))
        x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
        y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))        
        if w * h > 0 and x2 > x1 and y2 > y1:
            bbox = np.array([x1, y1, x2 - x1, y2 - y1])
        else:
            raise ValueError('bbox is invalid, x1={}, y1={}, x2={}, y2={}, w={}, h={}'.format(x1, y1, x2, y2, w, h))
        # aspect ratio preserving bbox
        w = bbox[2]
        h = bbox[3]
        c_x = bbox[0] + w / 2.
        c_y = bbox[1] + h / 2.
        aspect_ratio = cfg.IMG_RES[1] / cfg.IMG_RES[0]
        if w > aspect_ratio * h:
            h = w / aspect_ratio
        elif w < aspect_ratio * h:
            w = h * aspect_ratio
        bbox[2] = w
        bbox[3] = h
        bbox[0] = c_x - bbox[2] / 2.
        bbox[1] = c_y - bbox[3] / 2.

        bb_c_x = float(bbox[0] + 0.5*bbox[2])
        bb_c_y = float(bbox[1] + 0.5*bbox[3])
        bb_width = float(bbox[2])
        bb_height = float(bbox[3])

        src_w = bb_width * cfg.INTERHAND.SCALE_CROP
        src_h = bb_height * cfg.INTERHAND.SCALE_CROP
        src_center = np.array([bb_c_x, bb_c_y], np.float32)
        src_downdir = np.array([0, src_h*0.5], dtype=np.float32)
        src_rightdir = np.array([src_w*0.5, 0], dtype=np.float32)

        dst_w = cfg.IMG_RES[1]
        dst_h = cfg.IMG_RES[0]
        dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
        dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
        dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = src_center
        src[1, :] = src_center + src_downdir
        src[2, :] = src_center + src_rightdir

        dst = np.zeros((3, 2), dtype=np.float32)
        dst[0, :] = dst_center
        dst[1, :] = dst_center + dst_downdir
        dst[2, :] = dst_center + dst_rightdir

        trans_2x3 = cv.getAffineTransform(np.float32(src), np.float32(dst))
        trans_2x3 = trans_2x3.astype(np.float32)
        
        img_seg = cv.warpAffine(img_seg, trans_2x3, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_LINEAR)
        mask = cv.warpAffine(mask, trans_2x3, (cfg.IMG_RES[1], cfg.IMG_RES[0]), flags=cv.INTER_LINEAR)

        # update intrinsics correspondingly
        # Ref: https://stackoverflow.com/a/74753976
        K[0, 0] = K[0, 0] * cfg.IMG_RES[1] / (bbox[2] * cfg.INTERHAND.SCALE_CROP)
        K[1, 1] = K[1, 1] * cfg.IMG_RES[0] / (bbox[3] * cfg.INTERHAND.SCALE_CROP)
        K[:2, 2] = (trans_2x3 @ np.array([K[0, 2], K[1, 2], 1.0]))[:2]
        
        img_seg_with_alpha = np.zeros((img_seg.shape[0], img_seg.shape[1], 4), dtype=np.uint8)
        img_seg_with_alpha[:, :, :3] = img_seg
        img_seg_with_alpha[:, :, 3] = mask[:, :, 0]

        # ==============================================================================================
        # obtain mv and mvp transformation matrices from camera parameters
        # ==============================================================================================
        
        # nvdiffrast requires world to ndc transformation matrix
        # pytorch3d camera has utility functions to extract this info, so let's use it
        # we need to be careful about the camera/view space conventions (Ref: https://docs.nerf.studio/en/latest/quickstart/data_conventions.html#coordinate-conventions)
        # Camera/view space conventions:
        #   opengl/blender/nvdiffrast: [right, up, backward] (Ref: https://nvlabs.github.io/nvdiffrast/#coordinate-systems)
        #   opencv/colmap: [right, down, forward]
        #   pytorch3d: [left, up, forward] (Ref: https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md#camera-coordinate-systems)
        #   llff: [down, right, backward] (Ref: https://github.com/fyusion/llff#using-your-own-poses-without-running-colmap)
        world_to_cam_mtx_opencv = utils.create_4x4_trans_mat_from_R_t(R, t)
        opencv_to_opengl_mtx = utils.create_4x4_trans_mat_from_R_t(Rotation.from_euler("X", np.pi).as_matrix())
        world_to_cam_mtx_opengl = opencv_to_opengl_mtx @ world_to_cam_mtx_opencv
        mv = world_to_cam_mtx_opengl
        proj = utils.create_proj_mat_from_pinhole_camera(img_seg_with_alpha.shape[1], img_seg_with_alpha.shape[0], K[0, 0], K[1, 1], K[0, 2], K[1, 2], cfg.CAM_NEAR_FAR[0], cfg.CAM_NEAR_FAR[1])
        # image coordinates use opengl convention, so origin is at bottom left, +X is along right, +Y is along top
        # so invert top, bottom
        proj[1, 1] *= -1
        proj[1, 2] *= -1
        mvp = proj @ mv
        
        cv.imwrite(f"{out_img_raw_dir}/{i_data:05d}.png", cv.cvtColor(img_raw, cv.COLOR_RGB2BGR))
        cv.imwrite(f"{out_img_seg_dir}/{i_data:05d}.png", cv.cvtColor(img_seg_with_alpha, cv.COLOR_RGBA2BGRA))
        np.save(f"{out_mano_param_dir}/{i_data:05d}.npy", np.concatenate([shape, trans, pose]))
        np.save(f"{out_mano_verts_dir}/{i_data:05d}.npy", verts_world)
        
        np.save(f"{out_mv_dir}/{i_data:05d}.npy", mv)
        np.save(f"{out_mvp_dir}/{i_data:05d}.npy", mvp)

        i_data += 1
        
    subprocess.run(["iha/utils/create_video_from_frames.sh", "-f", "1", "-s", f"{0}", "-w", "%05d.png", f"{out_img_raw_dir}"])
    subprocess.run(["iha/utils/create_video_from_frames.sh", "-f", "1", "-s", f"{0}", "-w", "%05d.png", f"{out_img_seg_dir}"])



from pathlib import Path
import torch
from tqdm import tqdm
import cv2 as cv
import numpy as np

from ..render import util
from .. import utils as utils

logger = utils.get_logger(__name__)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, preprocess_dir, device, frame_skip=1):
        super().__init__()
        self.device = device

        list_path_to_img_seg = list(sorted(Path(f"{preprocess_dir}/img_seg").glob("*.png")))
        list_id_frame = range(0, len(list_path_to_img_seg), frame_skip)

        self.list_img = []
        self.list_mano_param = []
        self.list_mano_verts = []
        self.list_mano_offsets = []
        self.list_mv = []
        self.list_mvp = []
        self.list_campos = []
        self.list_id_data = []

        offsets_present = Path(f"{preprocess_dir}/mano_offsets").exists()
    
        for id_data, id_frame in enumerate(tqdm(list_id_frame, desc="Appending to list")):    
            self.list_id_data.append(id_data)

            img = cv.cvtColor(cv.imread(f"{preprocess_dir}/img_seg/{id_frame:05d}.png", cv.IMREAD_UNCHANGED), cv.COLOR_BGRA2RGBA)    # (H, W, 4)
            img = torch.from_numpy(img).float()/255
            img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
            self.list_img.append(img)

            mano_param = np.load(f"{preprocess_dir}/mano_param/{id_frame:05d}.npy")  # (10+3+45, )
            self.list_mano_param.append(torch.from_numpy(mano_param).float())

            mano_verts = np.load(f"{preprocess_dir}/mano_verts/{id_frame:05d}.npy")  # (778, 3)
            self.list_mano_verts.append(torch.from_numpy(mano_verts).float())
            
            if offsets_present:
                mano_offsets = np.load(f"{preprocess_dir}/mano_offsets/{id_frame:05d}.npy")  # (778, 3)
            else:
                mano_offsets = np.zeros_like(mano_verts)
            self.list_mano_offsets.append(torch.from_numpy(mano_offsets).float())

            mv = np.load(f"{preprocess_dir}/mv/{id_frame:05d}.npy")    # (4, 4)
            self.list_mv.append(torch.from_numpy(mv).float())

            mvp = np.load(f"{preprocess_dir}/mvp/{id_frame:05d}.npy")    # (4, 4)
            self.list_mvp.append(torch.from_numpy(mvp).float())

            campos = torch.linalg.inv(torch.from_numpy(mv).float())[:3, 3]    # (3,)
            self.list_campos.append(campos)

        self.num_data = len(self.list_id_data)
        
    def __len__(self):
        return self.num_data

    def __getitem__(self, i_data):
        return {
            "id_data": self.list_id_data[i_data],
            "img": self.list_img[i_data][None, :, :, :].to(self.device),
            "mano_param": self.list_mano_param[i_data][None, :].to(self.device),
            "mano_verts": self.list_mano_verts[i_data][None, :, :].to(self.device),
            "mano_offsets": self.list_mano_offsets[i_data][None, :, :].to(self.device),
            "mv": self.list_mv[i_data][None, :, :].to(self.device),
            "mvp": self.list_mvp[i_data][None, :, :].to(self.device),
            "campos": self.list_campos[i_data][None, :].to(self.device),
        }

    def collate(self, batch):
        out_batch = {
            'id_data' : [item['id_data'] for item in batch],
            'img' : torch.cat(list([item['img'] for item in batch]), dim=0),
            'mano_param' : torch.cat(list([item['mano_param'] for item in batch]), dim=0),
            'mano_verts' : torch.cat(list([item['mano_verts'] for item in batch]), dim=0),
            'mano_offsets' : torch.cat(list([item['mano_offsets'] for item in batch]), dim=0),
            'campos' : torch.cat(list([item['campos'] for item in batch]), dim=0),
            'mv' : torch.cat(list([item['mv'] for item in batch]), dim=0),
            'mvp' : torch.cat(list([item['mvp'] for item in batch]), dim=0),
        }
        return out_batch
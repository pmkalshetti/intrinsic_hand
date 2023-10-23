import numpy as np
from pathlib import Path
import logging
import cv2 as cv
import subprocess
import torch
import pickle
from tqdm import tqdm
import argparse

from iha.dataset.interhand import preprocess_interhand
from iha.dataset.dataset import Dataset
from iha.render import bilateral_denoiser, light, mesh, obj
from iha.optimizer.optimizer import IHAOptimizer
import iha.utils as utils
from configs.defaults import get_cfg_defaults

torch.manual_seed(0)
np.random.seed(0)

logger = utils.get_logger("iha", level=logging.INFO, root=True)


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config_file", help="path to config file")
    # args = parser.parse_args()

    cfg = get_cfg_defaults()
    # if args.config_file:
    #     cfg.merge_from_file(args.config_file)

    logger.info(f"Experiment: {cfg.EXPT_NAME}")
    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    if cfg.DATA.NAME == "interhand":
        cfg.INTERHAND.TRAIN_CAM_ID = cfg.INTERHAND.TRAIN_CAM_ID[3:]
        cfg.INTERHAND.TRAIN_FRAME_ID = cfg.INTERHAND.TRAIN_FRAME_ID[5:]
        train_preprocess_dir = f"{cfg.INTERHAND.PREPROCESS_DIR}/{cfg.IMG_RES[0]}_{cfg.IMG_RES[1]}/{cfg.INTERHAND.TRAIN_SPLIT}/capture_{cfg.INTERHAND.TRAIN_CAPTURE_ID}/{cfg.INTERHAND.TRAIN_SEQ_NAME}/cam_{cfg.INTERHAND.TRAIN_CAM_ID}/{cfg.INTERHAND.TRAIN_FRAME_ID}"
        if (not Path(train_preprocess_dir).exists()) or cfg.INTERHAND.FORCE_PREPROCESS:
            preprocess_interhand(cfg, cfg.INTERHAND.ROOT_DIR, cfg.INTERHAND.TRAIN_FRAMELIST_PATH if cfg.INTERHAND.TRAIN_FRAMELIST else False, cfg.INTERHAND.TRAIN_SPLIT, cfg.INTERHAND.TRAIN_CAPTURE_ID, cfg.INTERHAND.TRAIN_SEQ_NAME, cfg.INTERHAND.TRAIN_CAM_ID, cfg.INTERHAND.TRAIN_FRAME_ID, train_preprocess_dir)
        
        dataset_train = Dataset(train_preprocess_dir, cfg.DEVICE, cfg.INTERHAND.TRAIN_SKIP_FRAMES)
    else:
        raise NotImplementedError
   
    
    log_dir = f"{cfg.LOG.ROOT_DIR}/{cfg.EXPT_NAME}"; utils.create_dir(log_dir, True)
    with torch.cuda.device(cfg.DEVICE):
        ihaoptimizer = IHAOptimizer(cfg, dataset_train)
        ihaoptimizer.optimize(log_dir)
    
    out_opt_dir = f"{cfg.LOG.ROOT_DIR}/{cfg.EXPT_NAME}/opt_mesh_light"; utils.create_dir(out_opt_dir, True)
    ihaoptimizer.save_optimized_mesh_and_light(out_opt_dir)
    
if __name__ == "__main__":
    main()
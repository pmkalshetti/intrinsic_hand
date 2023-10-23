# this file is the one-stop reference point for all configurable options. It should be very well documented and provide sensible defaults for all options.
# Ref: https://github.com/rbgirshick/yacs#usage

from yacs.config import CfgNode as CN

_C = CN()

_C.EXPT_NAME = "iha_on_interhand"
_C.IMG_RES = [256, 256]
_C.DEVICE = "cuda:0"

# data
_C.DATA = CN()
_C.DATA.NAME = "interhand"

# Interhand
_C.INTERHAND = CN()
_C.INTERHAND.ROOT_DIR = "data/interhand"
_C.INTERHAND.PREPROCESS_DIR = f"output/interhand/preprocess"
_C.INTERHAND.SCALE_CROP = 1.3
_C.INTERHAND.FORCE_PREPROCESS = False

_C.INTERHAND.TRAIN_SPLIT = "test"
_C.INTERHAND.TRAIN_CAPTURE_ID = 1
_C.INTERHAND.TRAIN_SEQ_NAME = "ROM03_RT_No_Occlusion"
_C.INTERHAND.TRAIN_CAM_ID = "cam400270"
_C.INTERHAND.TRAIN_FRAME_ID = "frameall"
_C.INTERHAND.TRAIN_SKIP_FRAMES = 1
_C.INTERHAND.TRAIN_FRAMELIST = False
_C.INTERHAND.TRAIN_FRAMELIST_PATH = None

# MANO
_C.MANO = CN()
_C.MANO.PKL_PATH = "data/mano/models/MANO_RIGHT.pkl"
_C.MANO.HTML_UV = "data/html/TextureBasis/uvs_right.pkl"
_C.MANO.HTML_KD = "data/html/TextureSet/shadingRemoval/001_R.png"
_C.MANO.OBJ_DIR = "output/mano_obj"
_C.MANO.SUBDIVIDE = 2

# frustrum
_C.CAM_NEAR_FAR = [0.001, 1000.0] # near far plane in m

# render
_C.RENDER = CN()
_C.RENDER.PROBE_RES = 256  # env map probe resolution
_C.RENDER.N_SAMPLES = 12
_C.RENDER.DECORRELATED = False # use decorrelated sampling in forward and backward passes
_C.RENDER.DENOISER_DEMODULATE = True
_C.RENDER.SPP = 1
_C.RENDER.LAYERS = 1

# optimization
_C.BATCH_SIZE = 1 # DO NOT CHANGE THIS!
_C.OPT = CN() 
_C.OPT.LEARNING_RATE_LGT = 0.03
_C.OPT.LEARNING_RATE_MAT = 0.01
_C.OPT.LR_GEOM = 0.02
_C.OPT.LR_BETA = 0.0001
_C.OPT.LR_OFFSETS = 0.0001
_C.OPT.LR_HAND_POSE = 0.01
_C.OPT.LR_GLOBAL_ROT = 0.01
_C.OPT.LR_GLOBAL_TRANSL = 0.0001

_C.OPT.LAMBDA_KD = 0.1
_C.OPT.LAMBDA_KS = 0.05
_C.OPT.LAMBDA_NRM = 0.025
_C.OPT.LAMBDA_DIFFUSE = 0.15
_C.OPT.LAMBDA_SPECULAR = 0.0025
_C.OPT.LAMBDA_CHROMA = 0.025
_C.OPT.LAMBDA_NRM2 = 0.25
_C.OPT.LAPLACE = "relative"
_C.OPT.LAPLACE_SCALE = 1000.0
_C.OPT.LAMBDA_LPIPS = 0.09
_C.OPT.W_LGT_REG = 0.005

_C.OPT.EPOCHS = 100
_C.OPT.OPTIMIZE_LIGHT = True
_C.OPT.OPTIMIZE_MESH = True

_C.MAT = CN()
_C.MAT.KD_MIN = [0.03, 0.03, 0.03]
_C.MAT.KD_MAX = [0.8, 0.8, 0.8]
_C.MAT.KS_MIN = [0.0, 0.3, 0.0]
_C.MAT.KS_MAX = [0.0, 0.5, 0.0]
_C.MAT.NRM_MIN = [-1.0, -1.0, 0.0]
_C.MAT.NRM_MAX = [1.0, 1.0, 1.0]
# _C.MAT.TEXTURE_RES = [1024, 1024]
_C.MAT.TEXTURE_RES = [2048, 2048]
_C.MAT.NO_PERTURBED_NRM = False
_C.MAT.BSDF = 'pbr'

_C.LOG = CN()
_C.LOG.SAVE_INTERVAL = 10
_C.LOG.ROOT_DIR = f"output/log"


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
cfg = _C  # users can `from config import cfg`

# cfg.merge_from_file("configs/handavatar_comparison.yaml")

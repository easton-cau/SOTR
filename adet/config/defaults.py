from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False
_C.MODEL.BACKBONE.ANTI_ALIAS = False
_C.MODEL.RESNETS.DEFORM_INTERVAL = 1
_C.INPUT.HFLIP_TRAIN = True
_C.INPUT.CROP.CROP_INSTANCE = True


# ---------------------------------------------------------------------------- #
# VoVNet backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.VOVNET = CN()
_C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
_C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.VOVNET.NORM = "FrozenBN"
_C.MODEL.VOVNET.OUT_CHANNELS = 256
_C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256


# ---------------------------------------------------------------------------- #
# DLA backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.DLA = CN()
_C.MODEL.DLA.CONV_BODY = "DLA34"
_C.MODEL.DLA.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.DLA.NORM = "FrozenBN"


# ---------------------------------------------------------------------------- #
# Basis Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BASIS_MODULE = CN()
_C.MODEL.BASIS_MODULE.NAME = "ProtoNet"
_C.MODEL.BASIS_MODULE.NUM_BASES = 4
_C.MODEL.BASIS_MODULE.LOSS_ON = False
_C.MODEL.BASIS_MODULE.ANN_SET = "coco"
_C.MODEL.BASIS_MODULE.CONVS_DIM = 128
_C.MODEL.BASIS_MODULE.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.BASIS_MODULE.NORM = "SyncBN"
_C.MODEL.BASIS_MODULE.NUM_CONVS = 3
_C.MODEL.BASIS_MODULE.COMMON_STRIDE = 8
_C.MODEL.BASIS_MODULE.NUM_CLASSES = 80
_C.MODEL.BASIS_MODULE.LOSS_WEIGHT = 0.3


# ---------------------------------------------------------------------------- #
# TOP Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.TOP_MODULE = CN()
_C.MODEL.TOP_MODULE.NAME = "conv"
_C.MODEL.TOP_MODULE.DIM = 16


# ---------------------------------------------------------------------------- #
# BiFPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.BiFPN = CN()
# Names of the input feature maps to be used by BiFPN
# They must have contiguous power of 2 strides
# e.g., ["res2", "res3", "res4", "res5"]
_C.MODEL.BiFPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
_C.MODEL.BiFPN.OUT_CHANNELS = 160
_C.MODEL.BiFPN.NUM_REPEATS = 6

# Options: "" (no norm), "GN"
_C.MODEL.BiFPN.NORM = ""


# ---------------------------------------------------------------------------- #
# SOTR Options
# ---------------------------------------------------------------------------- #
_C.MODEL.SOTR = CN()

# Instance hyper-parameters
_C.MODEL.SOTR.INSTANCE_IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
_C.MODEL.SOTR.FPN_INSTANCE_STRIDES = [8, 8, 16, 32, 32]
_C.MODEL.SOTR.FPN_SCALE_RANGES = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
_C.MODEL.SOTR.SIGMA = 0.2
# Channel size for the instance head.
_C.MODEL.SOTR.INSTANCE_IN_CHANNELS = 256
_C.MODEL.SOTR.INSTANCE_CHANNELS = 512
# Convolutions to use in the instance head.
_C.MODEL.SOTR.NUM_INSTANCE_CONVS = 4
_C.MODEL.SOTR.USE_DCN_IN_INSTANCE = False
_C.MODEL.SOTR.TYPE_DCN = 'DCN'
_C.MODEL.SOTR.NUM_GRIDS = [40, 36, 24, 16, 12]
# Number of foreground classes.
_C.MODEL.SOTR.NUM_CLASSES = 80   # COCO
_C.MODEL.SOTR.NUM_KERNELS = 256
_C.MODEL.SOTR.NORM = "GN"
_C.MODEL.SOTR.USE_COORD_CONV = True
_C.MODEL.SOTR.PRIOR_PROB = 0.01

# Mask hyper-parameters.
# Channel size for the mask tower.
_C.MODEL.SOTR.MASK_IN_FEATURES = ["p2", "p3", "p4", "p5"]
_C.MODEL.SOTR.MASK_IN_CHANNELS = 256
_C.MODEL.SOTR.MASK_CHANNELS = 128
_C.MODEL.SOTR.NUM_MASKS = 256

# Test cfg.
# _C.MODEL.SOTR.CONFIDENCE_SCORE = 0.25
_C.MODEL.SOTR.NMS_PRE = 500
_C.MODEL.SOTR.SCORE_THR = 0.1
_C.MODEL.SOTR.UPDATE_THR = 0.05
_C.MODEL.SOTR.MASK_THR = 0.5
_C.MODEL.SOTR.MAX_PER_IMG = 100
_C.MODEL.SOTR.RESIZE_INPUT_FACTOR = 1
# NMS type: matrix OR mask.
_C.MODEL.SOTR.NMS_TYPE = "matrix"
# Matrix NMS kernel type: gaussian OR linear.
_C.MODEL.SOTR.NMS_KERNEL = "gaussian"
_C.MODEL.SOTR.NMS_SIGMA = 2

# Loss cfg.
_C.MODEL.SOTR.LOSS = CN()
_C.MODEL.SOTR.LOSS.FOCAL_USE_SIGMOID = True
_C.MODEL.SOTR.LOSS.FOCAL_ALPHA = 0.25
_C.MODEL.SOTR.LOSS.FOCAL_GAMMA = 2.0
_C.MODEL.SOTR.LOSS.FOCAL_WEIGHT = 1.0
_C.MODEL.SOTR.LOSS.DICE_WEIGHT = 3.0
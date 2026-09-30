"""Dimensions and defaults of the original Route B implementation."""


NSD_SUBJECTS = [f'subj{i:02d}' for i in range(1, 9)]
NUM_SUBJECTS = len(NSD_SUBJECTS)
COCO_SEARCH18_CATEGORIES_ALL = [
  'bottle',
  'bowl',
  'car',
  'chair',
  'clock',
  'cup',
  'fork',
  'keyboard',
  'knife',
  'laptop',
  'microwave',
  'mouse',
  'oven',
  'potted plant',
  'sink',
  'stop sign',
  'toilet',
  'tv'
]
COCO_SEARCH18_CATEGORIES = [
  'bottle',
  'car',
  'chair',
  'clock',
  'cup',
  'fork',
  'keyboard',
  'laptop',
  'microwave',
  'mouse',
  'oven',
  'potted plant',
  'sink',
  'stop sign',
  'toilet',
  'tv'
]
COCO_CAT_NAME_TO_ID = {
  'bottle': 44,
  'bowl': 51,
  'car': 3,
  'chair': 62,
  'clock': 85,
  'cup': 47,
  'fork': 48,
  'keyboard': 76,
  'knife': 49,
  'laptop': 73,
  'microwave': 78,
  'mouse': 74,
  'oven': 79,
  'potted plant': 64,
  'sink': 81,
  'stop sign': 13,
  'toilet': 70,
  'tv': 72
}
COCO_CAT_ID_TO_NAME = {v: k for (k, v) in COCO_CAT_NAME_TO_ID.items()}
DINO_MODEL_NAME = 'dinov2_vitl14'
DINO_EMBED_DIM = 1024
DINO_NUM_LAYERS = 24
CLIP_MODEL_NAME = 'ViT-L-14'
CLIP_PRETRAINED = 'laion2b_s32b_b82k'
CLIP_EMBED_DIM = 768
ROI_TIER_LABELS = {'early': [1], 'mid': [2, 3, 4], 'late': [5, 6, 7]}
DINO_TIER_LAYERS = {'early': list(range(0, 8)), 'mid': list(range(8, 16)), 'late': list(range(16, 24))}
PCA_COMPONENTS_PER_TIER = 2048
MAX_VOXELS = 39548
FMRI_ENCODER_HIDDEN = 2048
FMRI_ENCODER_LATENT = 2048
FMRI_ENCODER_DROPOUT = 0.4
SUBJECT_EMBED_DIM = 32
NUM_CATEGORIES = 16
CATEGORY_EMBED_DIM = 32
BATCH_SIZE = 256
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.01
NUM_EPOCHS_PRETRAIN = 100
NUM_EPOCHS_COMPLEXITY = 60
N_CV_FOLDS = 5
PATIENCE = 15
INFONCE_TEMPERATURE = 0.07
SEED = 42
NUM_GPUS = 4
NUM_WORKERS = 16

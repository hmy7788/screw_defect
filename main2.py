import sys
import torch

sys.path.append('.')
from src.utils import *
from src.visualization import *
from src.engine import *
from src.dataset import *

set_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 0  # Windows/Jupyter에서는 0이 가장 안정적
print(f"Device: {device} | num_workers: {NUM_WORKERS}")

run_dirs = make_run_dir('./outputs')

DATA_DIR = './data/new_k-fold_data/train'
CLASS_NAMES = ['good', 'type1', 'type2', 'type3', 'type4', 'type5']

all_paths, all_labels_np = get_image_paths_and_labels(DATA_DIR, CLASS_NAMES)

# BAD_WEIGHT_LIST = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

y_true_test, y_pred_test, model, history = train_test_split('resnet18', 3.5)
import os
import random
import torch
import numpy as np
from datetime import datetime

def set_seed(seed=42):
    """실험 재현성을 위해 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_dir(base_path='./outputs'):
    """학습을 돌릴 때마다 고유한 실험 폴더를 생성합니다."""
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(base_path, f"run_{now}")

    dirs = {
        'base': run_dir,
        'weights': os.path.join(run_dir, 'weights'),
        'figures': os.path.join(run_dir, 'figures')
    }

    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print(f"새로운 실험 결과 폴더가 생성되었습니다: {run_dir}")
    return dirs
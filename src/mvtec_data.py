import os
from sklearn.model_selection import train_test_split


def load_mvtec_screw(data_root='./data/screw', test_size=0.4, seed=42):
    TRAIN_DIR = os.path.join(data_root, 'train')
    TEST_DIR = os.path.join(data_root, 'test')

    # -----------------------------------------------------------------
    # 1. 정상 나사 이미지 path, type, mask 모으기
    good_records = []

    train_good_dir = os.path.join(TRAIN_DIR, 'good')
    test_good_dir = os.path.join(TEST_DIR, 'good')

    for train_file_name in os.listdir(train_good_dir):
        good_records.append(
            {
                'path': os.path.join(train_good_dir, train_file_name),
                'type': 'good',
                'mask': None,
                'label': 0
            }
        )

    for test_file_name in os.listdir(test_good_dir):
        good_records.append(
            {
                'path': os.path.join(test_good_dir, test_file_name),
                'type': 'good',
                'mask': None,
                'label': 0
            }
        )

    # ------------------------------------------------------------------
    # 2. 불량 나사 이미지 path, type, mask 모으기
    DEFECT_TYPES = ['manipulated_front', 'scratch_head', 'scratch_neck', 'thread_side', 'thread_top']
    GT_DIR = os.path.join(data_root, 'ground_truth')
    defect_records = []

    for DT in DEFECT_TYPES:
        test_bad_dir = os.path.join(TEST_DIR, DT)
    
        for file in os.listdir(test_bad_dir):
            mask_file = file.split('.')[0] + '_mask.png'
            mask_dir = os.path.join(GT_DIR, DT, mask_file)

            defect_records.append(
                {
                    'path': os.path.join(test_bad_dir, file),
                    'type': DT,
                    'mask': mask_dir,
                    'label': 1
                }
            )

    # ------------------------------------------------------------------
    # 3. train_records, test_records 클래스 stratify 비율로 나누기
    all_records = good_records + defect_records
    all_types = [r['type'] for r in all_records]

    train_records, test_records = train_test_split(
        all_records,
        test_size=test_size,
        random_state=seed,
        stratify=all_types
    )

    return train_records, test_records



'''
good_records = [
    {
        'path': ~,
        'type': 'good',
        'mask': None
    }, ...
]


defect_records = [
    {
        'path': 'data\\screw\\test\\manipulated_front\\000.png',
        'type': 'manipulated_front',
        'mask': 'data\\screw\\ground_truth\\manipulated_front\\000_mask.png'
    },
    {
        'path': 'data\\screw\\test\\manipulated_front\\001.png',
        'type': 'manipulated_front',
        'mask': 'data\\screw\\ground_truth\\manipulated_front\\001_mask.png'
    },
    ...
    # manipulated_front 24개 끝나면
    {
        'path': 'data\\screw\\test\\scratch_head\\000.png',
        'type': 'scratch_head',
        'mask': 'data\\screw\\ground_truth\\scratch_head\\000_mask.png'
    },
    ...
]
'''
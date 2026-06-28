import os

TRAIN_DIR = r'data\screw\train'
TEST_DIR = r'data\screw\test'

all_good_paths = []

train_good_dir = os.path.join(TRAIN_DIR, 'good')
test_good_dir = os.path.join(TEST_DIR, 'good')

for train_file_name in os.listdir(train_good_dir):
    all_good_paths.append(os.path.join(train_good_dir, train_file_name))

for test_file_name in os.listdir(test_good_dir):
    all_good_paths.append(os.path.join(test_good_dir, test_file_name))

# print(all_good_paths[0])
# print(os.path.exists(all_good_paths[0]))

# print(all_good_paths[-1])
# print(os.path.exists(all_good_paths[-1]))
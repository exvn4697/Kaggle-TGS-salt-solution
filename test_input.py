import os

INPUT_PATH = './input/'
TRAIN_PATH = INPUT_PATH + 'train/'
TEST_PATH = INPUT_PATH + 'test/'

TRAIN_IMAGES_PATH = TRAIN_PATH + 'images/'
TRAIN_MASKS_PATH = TRAIN_PATH + 'masks/'
TEST_IMAGES_PATH = TEST_PATH + 'images/'
print(len(os.listdir(TRAIN_IMAGES_PATH)), len(os.listdir(TRAIN_MASKS_PATH)), len(os.listdir(TEST_IMAGES_PATH)))
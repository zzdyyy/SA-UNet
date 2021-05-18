import os
import sys
import cv2
import numpy as np
from keras.callbacks import ModelCheckpoint
from glob import glob

from scipy.misc.pilutil import *
from util import *

dataset_name = sys.argv[1]
test_files = {
    'IDRiD': sorted(glob('../Patho-GAN/data/IDRiD/train_512/*.jpg')) +
             sorted(glob('../Patho-GAN/data/IDRiD/test_512/*.jpg')),
    'FGADR': sorted(glob('../Patho-GAN/data/FGADR/resized_512/*.jpg')),
    'retinal-lesions': sorted(glob('../Patho-GAN/data/retinal-lesions/resized_512/*.jpg')),
}[dataset_name]
test_data = []

desired_size=592  # network input size
for i in test_files:
    im = imread(i)

    # pad to fit input size

    old_size = im.shape[:2]  # old_size is in (height, width) format
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]

    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    color2 = [0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)


    test_data.append(cv2.resize(new_im, (desired_size, desired_size)))

test_data = np.array(test_data)


x_test = test_data.astype('float32') / 255.

x_test = np.reshape(x_test, (len(x_test), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format


from  SA_UNet import *
model=SA_UNet(input_size=(desired_size,desired_size,3),start_neurons=16,lr=1e-3,keep_prob=1,block_size=1)
model.summary()
weight="Model/DRIVE/SA_UNet.h5"

if os.path.isfile(weight): model.load_weights(weight)
model_checkpoint = ModelCheckpoint(weight, monitor='val_acc', verbose=1, save_best_only=True)

y_pred = model.predict(x_test, verbose=1)
y_pred= crop_to_shape(y_pred,(len(x_test),512,512,1))
y_pred_threshold = []
i=0
for y in y_pred:

    _, temp = cv2.threshold(y, 0.5, 1, cv2.THRESH_BINARY)
    y_pred_threshold.append(temp)
    y = y * 255
    # cv2.imwrite('TEST/%s_VS.png' % os.path.basename(test_files[i]).split('/')[0], y)
    print(test_files[i])

    cv2.imwrite('{}/{}_VS.png'.format(
        os.path.dirname(test_files[i]),
        os.path.basename(test_files[i]).split('.')[0]
        ), temp * 255)

    i+=1

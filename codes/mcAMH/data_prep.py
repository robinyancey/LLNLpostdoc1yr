import cv2
import numpy as np
import csv
import imageio
import os
import random


# todo: tiff images
# todo: convert directly from numpy to proper tensors,
# todo: put underscores in class names (when done with current tests),
# todo: remove extra class that generates in damage sub-classes (Damaged* forgot what it was)
# toto: remove randomiztion

def imageViewer(arr, data, attributes):
    print('Explore images and annotations...')
    im_size = arr[0].shape
    w = h = im_size[0]
    print('image size', im_size)
    num_images = arr.shape[0]
    print('number of images', num_images)

    attributes = ['ind'] + attributes[3:5]
    print('modified attributes', attributes)

    print('sample labels:\n', data[0:5])

    for i in range(num_images):

        all3 = np.concatenate((arr[i][:, :, 0], arr[i][:, :, 1], arr[i][:, :, 2]), axis=1)
        cv2.imshow('Class: ' + data[i][1] + '  ' + 'Type: ' + data[i][2], all3)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

def imageWriter(arr, data, classes, name, versions = 1, train_split = 0.85, val_split = 0.10):
    # 'arr' is the numpy array of images that Chris gave me
    # 'data' is the csv file that Chris gave me
    # 'classes' is the group/sub-group of classes that will be used for testing ('Binary', 'Damage', or 'Non-Damage' or 'Unlabeled')
    # 'name' is the name of the output directory to write to

    num_images = arr.shape[0]
    print('Number of images to split into classes folders:', num_images)
    isExist = os.path.exists(name)
    if not isExist:
        os.mkdir(name)
    inds = list(range(num_images))

    # I guess shuffling is done twice but that's ok.
    inds = random.sample(inds, num_images)

    train_num = num_images * train_split
    print('train number', round(train_num))
    val_num = num_images * val_split
    print('val number', round(val_num))
    for i in range(num_images):

        if i < train_num:
            path = name + '/train/'
        elif train_num <= i < (train_num + val_num):
            path = name + '/valid/'
        else:
            path = name + '/test/'

        if classes == 'Binary':
            # put into 'damage or 'non-damage' folders
            path = path + str(data[inds[i]][1]) + '/'

        elif classes == 'Damage':
            # put into mutli-class folders, if any (without confusing network with the mix inside the un-sub-label damage/non-damage)
            if data[inds[i]][2] == 'ISBE' or data[inds[i]][1] != 'Damage' or data[inds[i]][2] == 'Damaged RAM' or data[inds[i]][2] == 'Damage': # damaged RAM only has 1 example image
                continue
            path = path + str(data[inds[i]][2]) + '/'

        elif classes == 'Non-Damage':
            if data[inds[i]][2] == 'AMP Scratch' or data[inds[i]][2] == 'Metallic Particle' or data[inds[i]][2] == 'NonMetallic Particle' or data[inds[i]][2] == 'TBD':
                path = path + str(data[inds[i]][2]) + '/'
            else:
                continue

        elif classes == 'Unlabeled':
            if data[inds[i]][2] == 'Non-Damage':
                path = path + 'test/' + str(data[inds[i]][2]) + '/'
            elif data[inds[i]][2] == 'Damage':
                path = path + 'test/' + str(data[inds[i]][2]) + '/'
            else:
                continue

        else:
            # put into mutli-class folders without any damage/non-damage separation (not reccomended)
            path = path + str(data[inds[i]][2]) + '/'

        isExist = os.path.exists(path)
        if not isExist:
            # make the folders for the image to go into
            os.makedirs(path)
        # full path for image w/ name
        path2 = path + '/' + str(data[inds[i]][2]) + '_' + str(inds[i]) + '_' + str(num_images)[0]

        print('Writing 4 images for original image #', inds[i], 'of', num_images)
        cv2.imwrite(path2 + '.jpg', arr[inds[i]])
        if versions == 4:
            for j in range(3):
                path3 = path2 + '_' + str(j)
                cv2.imwrite(path3 + '.jpg', arr[inds[i]][:, :, j])

    return

from collections import defaultdict

def count_classes(data):
    num_images = data.shape[0]
    damage = defaultdict(int)
    non_damage = defaultdict(int)
    for i in range(num_images):
        if data[i][1] == 'Damage':
            damage[str(data[i][2])] += 1
        elif data[i][1] == 'Non-Damage':
            non_damage[str(data[i][2])] += 1
    print('Damage')
    print({k: v for k, v in sorted(damage.items(), key=lambda item: item[1])})
    print('Non-damage')
    print({k: v for k, v in sorted(non_damage.items(), key=lambda item: item[1])})
    return



if __name__ == '__main__':
    # load images
    ims = np.load('/Users/yancey5/Desktop/all-records-test/array.npy')
    print('original number of images 1: ', ims.shape[0])

    ims2 = np.load('/Users/yancey5/Desktop/all-records-test/array2.npy')
    print('original number of images 2: ', ims2.shape[0])
    print(ims.shape) # double check h/w dims
    #print(ims[0][0]) # double check element vals
    # load labels
    with open('/Users/yancey5/Desktop/all-records-test/metadata.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        attributes = next(reader)
        data = np.array(list(reader))

    with open('/Users/yancey5/Desktop/all-records-test/metadata2.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        attributes = next(reader)
        data2 = np.array(list(reader))
    ims = ims[:25] # testing purposes
    ims2 = ims2[:25]
    #test = np.random.randint(0, 256, (155, 512, 512, 3)) # testing purposes
    #print(data[:3, 4:6])
    #print(data[:5, ])
    data = data[:, [6, 3, 4]]
    #print(data.shape)

    data2 = data2[:, [6, 3, 4]]
    #print(data2.shape)

    print('Features: ', attributes)

    count_classes(data)
    count_classes(data2)
    #imageWriter(ims, data, 'Binary', 'AMH_test44k')


    #imageWriter(ims, data, 'Binary', 'AMH_55k-Binary-mini', train_split = 0.80, val_split = 0)
    #imageWriter(ims2, data2, 'Binary', 'AMH_55k-Binary-mini', train_split = 0.80, val_split = 0)

    #imageViewer(ims, data, attributes)







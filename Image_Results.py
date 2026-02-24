import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def Image_Results():
    Pre = np.load('Pre_Images.npy', allow_pickle=True)
    Post = np.load('Post_Images.npy', allow_pickle=True)
    Det = np.load('Detected_Images.npy', allow_pickle=True)

    I = [46, 93, 114, 123, 222]
    for j in range(5):
        pre = Pre[I[j]]
        post = Post[I[j]]
        det = Det[I[j]]

        fig, ax = plt.subplots(1, 3)
        plt.suptitle(" Image %d" % ((j + 1)), fontsize=20)
        plt.subplot(1, 3, 1)
        plt.title('Pre Images')
        plt.imshow(pre)
        plt.subplot(1, 3, 2)
        plt.title('Post Images')
        plt.imshow(post)
        plt.subplot(1, 3, 3)
        plt.title('Change Detection')
        plt.imshow(det)
        plt.show()

        cv.imwrite('./Results/Image Results/PreImage-' + str(j + 1) + '.png', pre)
        cv.imwrite('./Results/Image Results/PostImage-' + str(j + 1) + '.png', post)
        cv.imwrite('./Results/Image Results/ChangeDetection-' + str(j + 1) + '.png', det)


if __name__ == '__main__':
    Image_Results()

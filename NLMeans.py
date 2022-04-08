import numpy as np
import matplotlib.pyplot as plt
from math import exp

def get_f_p_q(p, q, B_p, h):
    euclidean = (B_p[q] - B_p[p]) ** 2
    return exp(euclidean / h / h)


def NLmeans(noisy_img, window_size, h):
    '''

    :param noisy_img: Image with noise added. Assume image is already padded with boundary pixels
            flipped across the axis being padded
    :param window_size: Size of neighborhood to use
    :param h: hyper parameter for Gaussian weighting function
    :return: out, a denoised image
    '''

    R_p = window_size ** 2
    B_p = np.zeros(shape=noisy_img.shape)
    i_range, j_range = ((window_size // 2, noisy_img.shape[0] - window_size // 2), (window_size // 2, noisy_img.shape[1] - window_size // 2))
    orig_img = noisy_img[i_range[0]:i_range[1], j_range[0], j_range[1]]
    out = np.zeros(orig_img.shape)
    for i in range(i_range[0], i_range[1], 1):
        lower_i = i - window_size // 2
        upper_i = i + window_size // 2
        for j in range(j_range[0], j_range[1], 1):
            lower_j = j - window_size // 2
            upper_j = j + window_size // 2
            B_p[i, j] = np.sum(noisy_img[lower_i:upper_i, lower_j:upper_j]) / R_p

    for i in range(i_range[0], i_range[1], 1):
        lower_i = i - window_size // 2
        upper_i = i + window_size // 2
        for j in range(j_range[0], j_range[1], 1):
            lower_j = j - window_size // 2
            upper_j = j + window_size // 2
            p = (i - window_size // 2, j - window_size // 2)

            F_p = np.zeros(shape=orig_img.shape)
            for f_i in range(i_range[0], i_range[1], 1):
                for c_j in range(j_range[0], j_range[1], 1):
                    F_p[f_i - window_size // 2, c_j - window_size // 2] = get_f_p_q(p, (f_i, c_j), B_p, h)
            
            C_p = np.sum(F_p)
            vqFpq = np.multiply(orig_img, F_p)
            out[lower_i, lower_j] = np.sum(vqFpq) / C_p

    return out

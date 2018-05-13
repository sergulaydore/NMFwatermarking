"""
This module includes methods required for watermarking game
"""
import numpy as np
from skimage.transform import rotate
import importlib
import pywt
import utils
importlib.reload(utils)

eta = 1e-3
iterno =1000


def encoding(orgimg, rectnumber, rectsize, k, wmpower =0.9):
    """
    Encodes the watermark in H domain
    :param orgimg: np.array, original image
    :param rectnumber: int, number of rectangles
    :param rectsize: int, rectangle size
    :param k: int, rank of
    :return: hugemat, Wvec, m_begin_vec, m_finish_vec, n_begin_vec, n_finish_vec
    """

    m, n = orgimg.shape

    m_begin_vec = np.zeros(rectnumber)
    m_finish_vec = np.zeros(rectnumber)
    n_begin_vec = np.zeros(rectnumber)
    n_finish_vec = np.zeros(rectnumber)

    hugemat = np.zeros((k * rectnumber * rectsize, m * n))
    orgrect = np.zeros((rectsize, rectsize, rectnumber))
    Q = np.zeros((k, rectsize, rectnumber))

    WMvec = []
    Wlist = []

    # inside for loop
    for rectindex in range(rectnumber):

        W = np.random.uniform(size=(rectsize, k))
        Wlist.append(W)

        ###############################
        # generation PR rectangles
        m_begin_vec[rectindex] = np.random.randint(m - rectsize);
        m_finish_vec[rectindex] = m_begin_vec[rectindex] + rectsize;
        n_begin_vec[rectindex] = np.random.randint(n - rectsize);
        n_finish_vec[rectindex] = n_begin_vec[rectindex] + rectsize;

        ###############################
        # the rectangle we deal with
        # orgrect[:,:,rectindex]
        orgrect[:, :, rectindex] = orgimg[int(m_begin_vec[rectindex]):int(m_finish_vec[rectindex]),
                                   int(n_begin_vec[rectindex]):int(n_finish_vec[rectindex])
                                   ]

        ###############################
        # Watermark
        WM = np.sqrt(3 * wmpower) * np.random.randn(k, rectsize)  # Watermark to be embedded on H domain

        ###############################
        # Inverse NMF
        I = np.eye(k);
        K = I - eta * np.matmul(np.transpose(W), W)

        sum_K = np.zeros((k, k));
        for i in range(iterno):
            sum_K = sum_K + np.linalg.matrix_power(K, i)

        Q[:, :, rectindex] = eta * np.matmul(sum_K, np.transpose(W))

        # columnvec = np.linspace(int(n_begin_vec[rectindex]), int(n_finish_vec[rectindex]),
        #                         num=rectsize, endpoint=False, dtype=int)

        rowvec = np.linspace(int(m_begin_vec[rectindex]), int(m_finish_vec[rectindex]),
                             num=rectsize, endpoint=False, dtype=int)

        for columnindex in range(rectsize):
            mindex = rowvec[0]

            tempvec = np.where(orgimg.flatten('F') == orgrect[0, columnindex, rectindex])[0]

            if len(tempvec) > 1:
                temp = tempvec[np.mod(tempvec, m) == mindex]
            elif len(tempvec) == 1:
                temp = tempvec

            if len(temp) > 0:
                hugemat[
                (rectindex) * rectsize * k + (columnindex) * k:(rectindex) * rectsize * k + (columnindex) * k + k,
                temp[0]:temp[0] + rectsize] = Q[:, :, rectindex]  # DOUBLE CHECK!!!
            else:
                print("len(temp) == 0 should not happen")

        WMvec = np.concatenate((WMvec, WM.flatten("F")))

    normm = np.linalg.norm(WMvec)
    WMvec = WMvec / normm  # Normalizing the frobenious norm of watermark

    return hugemat, WMvec, m_begin_vec, m_finish_vec, n_begin_vec, n_finish_vec, Wlist

def embed_watermark(hugemat, orgimg, WMvec, PSNR=30):
    m, n = orgimg.shape
    [U, S, Vt] = np.linalg.svd(hugemat, full_matrices=False)
    V = np.transpose(Vt)
    r = np.linalg.matrix_rank(hugemat)
    SS = np.diag(1 / S[:r])
    Xdelta = np.matmul(V[:, :r], np.matmul(np.matmul(SS, np.transpose(U[:, :r])), WMvec))
    alpha = 255 * np.sqrt(m * n) / (np.linalg.norm(Xdelta)) * 10 ** (-PSNR / 20)
    WMvec = alpha * WMvec
    Xdelta = alpha * Xdelta
    orgimgwm = orgimg + np.reshape(Xdelta, orgimg.shape, order='F')
    return WMvec, orgimgwm

def attack_rotate(angle, orgimgwm, orgimg):
    noisyimgwm = rotate(orgimgwm, angle)  # Attacked WMked Image
    noisyimg = rotate(orgimg, angle)  # Attacked NOT WMked Image
    return noisyimgwm, noisyimg

def attack_rotate_v2(angle, orgimgspatial, Xdelta):
    noisyspatial = rotate(orgimgspatial, angle)
    wp = pywt.WaveletPacket2D(data=noisyspatial, wavelet='db4')
    noisyimg = wp['aaa'].data   # Attacked NOT WMked Image
    noisyimgwm = noisyimg + np.reshape(Xdelta, noisyimg.shape, order='F')  # Attacked WMked Image
    return noisyimgwm, noisyimg

def decoding(orgimg, orgimgwm, noisyimgwm, noisyimg, rectnumber, rectsize, m_begin_vec, Wlist, m_finish_vec, n_begin_vec, n_finish_vec, k):
    Hvec = []
    Hwmvec = []
    Hwmnoisyvec = []
    Hnoisyvec = []

    for rectangle_index in range(rectnumber):
        W = Wlist[rectangle_index]  # np.random.uniform(size=(rectsize, k))

        mstart, mfinish = int(m_begin_vec[rectangle_index]), int(m_finish_vec[rectangle_index])
        nstart, nfinish = int(n_begin_vec[rectangle_index]), int(n_finish_vec[rectangle_index])

        H, cost_list = utils.modified_nmf(orgimg[mstart:mfinish, nstart:nfinish], eta, iterno, k, W)
        Hwm, _ = utils.modified_nmf(orgimgwm[mstart:mfinish, nstart:nfinish], eta, iterno, k, W)
        Hwmnoisy, _ = utils.modified_nmf(noisyimgwm[mstart:mfinish, nstart:nfinish], eta, iterno, k, W)
        Hnoisy, _ = utils.modified_nmf(noisyimg[mstart:mfinish, nstart:nfinish], eta, iterno, k, W)

        Hvec = np.concatenate((Hvec, H.flatten("F")))
        Hwmvec = np.concatenate((Hwmvec, Hwm.flatten("F")))
        Hwmnoisyvec = np.concatenate((Hwmnoisyvec, Hwmnoisy.flatten("F")))
        Hnoisyvec = np.concatenate((Hnoisyvec, Hnoisy.flatten("F")))

    return Hvec, Hwmvec, Hwmnoisyvec, Hnoisyvec


def get_corrcoef(WMvec, Hvec, Hwmvec, Hwmnoisyvec, Hnoisyvec):
    Cresult_SB = np.corrcoef(Hwmvec - Hvec, WMvec.reshape(-1, ))[0][1]
    Cwmnoisy_SB = np.corrcoef(Hwmnoisyvec - Hvec, WMvec.reshape(-1, ))[0][1]
    Cnoisy_SB = np.corrcoef(Hnoisyvec - Hvec, WMvec.reshape(-1, ))[0][1]
    return Cresult_SB, Cwmnoisy_SB, Cnoisy_SB

def run(orgimg, rectsize, rectnumber, k, rotation=None):
    hugemat, WMvec, m_begin_vec, m_finish_vec, n_begin_vec, n_finish_vec, Wlist = encoding(orgimg, rectnumber,
                                                                                                 rectsize, k)
    WMvec, orgimgwm = embed_watermark(hugemat, orgimg, WMvec, PSNR=30)
    if rotation!=None:
        noisyimgwm, noisyimg = attack_rotate(rotation, orgimgwm, orgimg)
    Hvec, Hwmvec, Hwmnoisyvec, Hnoisyvec = decoding(orgimg, orgimgwm, noisyimgwm, noisyimg, rectnumber,
                                                                 rectsize, m_begin_vec, Wlist, m_finish_vec, n_begin_vec,
                                                                 n_finish_vec, k)
    Cresult_SB, Cwmnoisy_SB, Cnoisy_SB = get_corrcoef(WMvec, Hvec, Hwmvec, Hwmnoisyvec, Hnoisyvec)
    return Cresult_SB, Cwmnoisy_SB, Cnoisy_SB




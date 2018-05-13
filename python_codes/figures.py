from sklearn.metrics import roc_curve, auc
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pywt
from scipy import misc
import importlib
import watermarking
importlib.reload(watermarking)

#############################################################################################
rectsize = 20
rectnumber = 5
k = 2
N = 1000

plt.figure()
angle_list = [0, 0.5, 1, 1.5, 2, 2.5, 3]
for angle in angle_list:
    filename = "corrcoefs_angle_" + str(angle) + "_rectsize_" + str(rectsize) + "_rectnumber_" + str(rectnumber) + "_k_" + str(k) + ".pkl"

    with open("./results/" + filename, 'rb') as f:
        Cresult_SB_list, Cwmnoisy_SB_list, Cnoisy_SB_list = pickle.load(f)

    y_true = np.concatenate((np.zeros(N), np.ones(N)))
    y_scores = np.concatenate((Cnoisy_SB_list, Cwmnoisy_SB_list))

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    print(angle, roc_auc)

    lw = 2
    plt.plot(fpr, tpr, lw=lw, label=' angle = %0.2f' % angle)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.grid(True)

plt.legend()
plt.show()

#############################################################################################
angle = 1
rectsize = 20
rectnumber = 5
k = 2
filename = "corrcoefs_angle_" + str(angle) + "_rectsize_" + str(rectsize) + "_rectnumber_" + str(
    rectnumber) + "_k_" + str(k) + ".pkl"

with open("./results/" + filename, 'rb') as f:
    Cresult_SB_list, Cwmnoisy_SB_list, Cnoisy_SB_list = pickle.load(f)

# the histogram of the data
binsize = 32
plt.figure()
# n, bins, patches = plt.hist(Cresult_SB_list, binsize, normed=1, facecolor='green', alpha=0.75, label="Without attack")
n, bins, patches = plt.hist(Cwmnoisy_SB_list, binsize, normed=1, facecolor='blue', alpha=0.75, label="With attack on watermarked image")
n, bins, patches = plt.hist(Cnoisy_SB_list, binsize, normed=1, facecolor='red', alpha=0.75, label="With attack on unwatermarked image")

# # add a 'best fit' line
# y = mlab.normpdf(bins, np.mean(Cresult_SB_list), np.std(Cresult_SB_list))
# l = plt.plot(bins, y, 'r--', linewidth=1)

plt.xlabel('Correlation Coefficient in H domain', fontsize=14)
plt.ylabel('Normalized Frequency', fontsize=14)
plt.legend()
# plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
# plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

#############################################################################################
lena = misc.imread("./images/im_512_3.tif")
wp = pywt.WaveletPacket2D(data=lena, wavelet='db4')
orgimg = wp['aa'].data  #wp['aa'].data  # (lena[:100, :100]).astype(float) # wp['aaa'].data
rectnumber, rectsize = 5, 15

hugemat, WMvec, m_begin_vec, m_finish_vec, n_begin_vec, n_finish_vec, Wlist = watermarking.encoding(orgimg, rectnumber, rectsize, k)
# Displaying rectangles
orgimg_squared=orgimg.copy()
for rectindex in range(rectnumber):
    orgimg_squared[int(m_begin_vec[rectindex]),int(n_begin_vec[rectindex]):int(n_finish_vec[rectindex])]=0;
    orgimg_squared[int(m_finish_vec[rectindex]),int(n_begin_vec[rectindex]):int(n_finish_vec[rectindex])]=0;
    orgimg_squared[int(m_begin_vec[rectindex]):int(m_finish_vec[rectindex]),int(n_begin_vec[rectindex])]=0;
    orgimg_squared[int(m_begin_vec[rectindex]):int(m_finish_vec[rectindex]),int(n_finish_vec[rectindex])]=0;

fig = plt.imshow(orgimg_squared, cmap='gray')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()
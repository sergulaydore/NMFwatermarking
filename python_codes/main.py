import pywt
from scipy import misc
import numpy as np
import importlib
import pickle
import watermarking
importlib.reload(watermarking)

lena = misc.imread("./images/im_512_3.tif")
wp = pywt.WaveletPacket2D(data=lena, wavelet='db4')
orgimg = wp['aa'].data # (lena[:100, :100]).astype(float) # wp['aaa'].data
rectsize = 20
rectnumber = 5
k = 2
N = 1000

angle_list = [0, 0.5, 1, 1.5, 2, 2.5, 3]

for angle in angle_list:
    Cresult_SB_list = np.zeros(N)
    Cwmnoisy_SB_list = np.zeros(N)
    Cnoisy_SB_list = np.zeros(N)
    print("angle=", angle)
    for i in range(N):
        if i % 100 == 0:
            print("Experiment", i)
        Cresult_SB_list[i], Cwmnoisy_SB_list[i], Cnoisy_SB_list[i] = watermarking.run(orgimg, rectsize, rectnumber, k, angle)

    filename = "corrcoefs_angle_" + str(angle) + "_rectsize_" + str(rectsize) + "_rectnumber_" + str(rectnumber) + "_k_" + str(k) + ".pkl"
    with open("./results/" + filename, 'wb') as f:
        pickle.dump([Cresult_SB_list, Cwmnoisy_SB_list, Cnoisy_SB_list], f)

# # # Getting back the objects:
# # with open("./results/" + filename, 'rb') as f:
# #     a, b, c = pickle.load(f)
#
# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt
#
# # the histogram of the data
# binsize = 16
# plt.figure()
# n, bins, patches = plt.hist(Cresult_SB_list, binsize, normed=1, facecolor='green', alpha=0.75, label="Without attack")
# n, bins, patches = plt.hist(Cwmnoisy_SB_list, binsize, normed=1, facecolor='blue', alpha=0.75, label="With attack watermarked")
# n, bins, patches = plt.hist(Cnoisy_SB_list, binsize, normed=1, facecolor='red', alpha=0.75, label="With attack unwatermarked")
#
# # # add a 'best fit' line
# # y = mlab.normpdf(bins, np.mean(Cresult_SB_list), np.std(Cresult_SB_list))
# # l = plt.plot(bins, y, 'r--', linewidth=1)
#
# plt.xlabel('Correlation Coefficient in H domain')
# plt.ylabel('Probability')
# plt.legend()
# # plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
# # plt.axis([40, 160, 0, 0.03])
# plt.grid(True)
# plt.show()
#
# from sklearn.metrics import roc_curve, auc
# y_true = np.concatenate((np.zeros(N), np.ones(N)))
# y_scores = np.concatenate((Cnoisy_SB_list, Cwmnoisy_SB_list))
#
# fpr, tpr, _ = roc_curve(y_true, y_scores)
# roc_auc = auc(fpr, tpr)
#
# plt.figure()
# lw = 2
# plt.plot(fpr, tpr, color='darkorange',
#          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.grid(True)
# # plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()

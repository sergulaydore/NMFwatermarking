import pywt
from scipy import misc
import matplotlib.pyplot as plt
import importlib
import watermarking
importlib.reload(watermarking)

lena = misc.imread("./images/im_512_3.tif")
wp = pywt.WaveletPacket2D(data=lena, wavelet='db4')
orgimg = wp['aa'].data # (lena[:100, :100]).astype(float) # wp['aaa'].data
rectsize = 40
rectnumber = 10
k = 2

hugemat, WMvec, m_begin_vec, m_finish_vec, n_begin_vec, n_finish_vec, Wlist = watermarking.encoding(orgimg, rectnumber, rectsize, k)
WMvec, orgimgwm = watermarking.embed_watermark(hugemat, orgimg, WMvec, PSNR=30)
noisyimgwm, noisyimg = watermarking.attack_rotate(1, orgimgwm, orgimg)
Hvec, Hwmvec, Hwmnoisyvec, Hnoisyvec = watermarking.decoding(orgimg, orgimgwm, noisyimgwm, noisyimg, rectnumber,
                                                             rectsize, m_begin_vec, Wlist, m_finish_vec, n_begin_vec,
                                                             n_finish_vec, k)
Cresult_SB, Cwmnoisy_SB, Cnoisy_SB = watermarking.get_corrcoef(WMvec, Hvec, Hwmvec, Hwmnoisyvec, Hnoisyvec)

print("Correlation between original image and unattacked watermarked image is", Cresult_SB)
print("Correlation between original image and attacked watermarked image is", Cwmnoisy_SB)
print("Correlation between original image and attacked but none-watermarked image is", Cnoisy_SB)

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
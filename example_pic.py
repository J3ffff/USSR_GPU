import cv2
import numpy as np
import matplotlib.pyplot as plt

img_1_hr = cv2.imread('./Images/example/img_004_SRF_3_HR.png')
img_1_bicubic = cv2.imread('./Images/example/img_004_SRF_3_Bicubic.png')
img_1_selfexsr= cv2.imread('./Images/example/img_004_SRF_3_SelfExSR.png')
img_1_ussr = cv2.imread('./Images/example/img_004_SRF_3_USSR.png')



img_1_hr_roi = img_1_hr[110:161, 100:151]
img_1_bicubic_roi = img_1_bicubic[110:161, 100:151]
img_1_selfexsr_roi = img_1_selfexsr[110:161, 100:151]
img_1_ussr_roi = img_1_ussr[110:161, 100:151]

img_2_hr = cv2.imread('./Images/example/img_009_SRF_3_HR.png')
img_2_bicubic = cv2.imread('./Images/example/img_009_SRF_3_Bicubic.png')
img_2_selfexsr= cv2.imread('./Images/example/img_009_SRF_3_SelfExSR.png')
img_2_ussr = cv2.imread('./Images/example/img_009_SRF_3_USSR.png')

img_2_hr_roi = img_2_hr[50:151, 50:151]
img_2_bicubic_roi = img_2_bicubic[50:151, 50:151]
img_2_selfexsr_roi = img_2_selfexsr[50:151, 50:151]
img_2_ussr_roi = img_2_ussr[50:151, 50:151]

img_3_hr = cv2.imread('./Images/example/img_002_SRF_4_HR.png')
img_3_bicubic = cv2.imread('./Images/example/img_002_SRF_4_Bicubic.png')
img_3_selfexsr= cv2.imread('./Images/example/img_002_SRF_4_SelfExSR.png')
img_3_ussr = cv2.imread('./Images/example/img_002_SRF_4_USSR.png')

img_3_hr_roi = img_3_hr[180:341, 370:531]
img_3_bicubic_roi = img_3_bicubic[180:341, 370:531]
img_3_selfexsr_roi = img_3_selfexsr[180:341, 370:531]
img_3_ussr_roi = img_3_ussr[180:341, 370:531]

# cv2.imwrite('./Images/example/img_004_SRF_3_HR_ROI.png',img_1_selfexsr_roi)
# cv2.imwrite('./Images/example/img_004_SRF_3_Bicubic_ROI.png',img_1_bicubic_roi)
# cv2.imwrite('./Images/example/img_004_SRF_3_SelfExSR_ROI.png',img_1_hr_roi)
# cv2.imwrite('./Images/example/img_004_SRF_3_USSR_ROI.png',img_1_ussr_roi)
# #
# cv2.imwrite('./Images/example/img_009_SRF_3_HR_ROI.png',img_2_hr_roi)
# cv2.imwrite('./Images/example/img_009_SRF_3_Bicubic_ROI.png',img_2_bicubic_roi)
# cv2.imwrite('./Images/example/img_009_SRF_3_SelfExSR_ROI.png',img_2_selfexsr_roi)
# cv2.imwrite('./Images/example/img_009_SRF_3_USSR_ROI.png',img_2_ussr_roi)

cv2.imwrite('./Images/example/img_002_SRF_4_HR_ROI.png',img_3_hr_roi)
cv2.imwrite('./Images/example/img_002_SRF_4_Bicubic_ROI.png',img_3_bicubic_roi)
cv2.imwrite('./Images/example/img_002_SRF_4_SelfExSR_ROI.png',img_3_selfexsr_roi)
cv2.imwrite('./Images/example/img_002_SRF_4_USSR_ROI.png',img_3_ussr_roi)

cv2.rectangle(img_1_hr, (100,110), (150,160),(0,0,255), 2)
cv2.rectangle(img_1_bicubic, (100,110), (150,160),(0,0,255), 2)
cv2.rectangle(img_1_selfexsr, (100,110), (150,160),(0,0,255), 2)
cv2.rectangle(img_1_ussr, (100,110), (150,160),(0,0,255), 2)
# cv2.imwrite('./Images/example/img_004_SRF_3_HR_BBox.png', img_1_hr)
# cv2.imwrite('./Images/example/img_004_SRF_3_Bicubic_BBox.png', img_1_bicubic)
# cv2.imwrite('./Images/example/img_004_SRF_3_SelfExSR_BBox.png', img_1_selfexsr)
# cv2.imwrite('./Images/example/img_004_SRF_3_USSR_BBox.png', img_1_ussr)

cv2.rectangle(img_2_hr, (50,50), (150,150), (0,0,255), 2)
cv2.rectangle(img_2_bicubic, (50,50), (150,150), (0,0,255), 2)
cv2.rectangle(img_2_selfexsr, (50,50), (150,150), (0,0,255), 2)
cv2.rectangle(img_2_ussr, (50,50), (150,150), (0,0,255), 2)
# cv2.imwrite('./Images/example/img_009_SRF_3_HR_BBox.png', img_2_hr)
# cv2.imwrite('./Images/example/img_009_SRF_3_Bicubic_BBox.png', img_2_bicubic)
# cv2.imwrite('./Images/example/img_009_SRF_3_SelfExSR_BBox.png', img_2_selfexsr)
# cv2.imwrite('./Images/example/img_009_SRF_3_USSR_BBox.png', img_2_ussr)

cv2.rectangle(img_3_hr, (370,180),(530,340),(0,0,255), 4)
cv2.imwrite('./Images/example/img_002_SRF_4_HR_BBox.png', img_3_hr)



plt.subplot(221)
plt.imshow(img_3_hr)
plt.subplot(222)
plt.imshow(img_3_bicubic_roi)
plt.subplot(223)
plt.imshow(img_3_selfexsr_roi)
plt.subplot(224)
plt.imshow(img_3_ussr_roi)
plt.show()
# cv2.imsh ow('', img_1_bicubic)
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


imgA = cv.imread("imageA.png")
imgB = cv.imread("imageB.png")


#Using ORB to get the features from the images

orb = cv.ORB_create(nfeatures=1000)


key_ptA, des1 = orb.detectAndCompute(imgA, None)
key_ptB, des2 = orb.detectAndCompute(imgB, None)

des1 = np.float32(des1)
des2 = np.float32(des2)

# Using FLANN algorithm and KNN matching to findout the matching features in both the images
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) 
flann = cv.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)
line_features = cv.drawMatchesKnn(imgA,key_ptA,imgB,key_ptB,matches,None)

src_pts = np.array([key_ptA[m[0].queryIdx].pt for m in matches])
dst_pts = np.array([key_ptB[m[0].trainIdx].pt for m in matches])

#finding homography and warping the images

H, __ = cv.findHomography(dst_pts, src_pts, cv.RANSAC, 4.0)

w, h,_ = imgA.shape

# print(imgA.shape)

imgB_corners = np.array([[0, 0, 1], [0, w, 1], [h, w, 1], [h, 0, 1]])
warped_corners = H @ imgB_corners.T


warped_corners = np.int0(np.round(warped_corners/warped_corners[2]))

# print(warped_corners)
h2 = warped_corners[0, -1]
# print(h2)
warped_image = cv.warpPerspective(imgB, H, (h2, w))

imgres = np.zeros((w, h2, 3), np.uint8)
imgres[:, :h , :] = imgA

cv.fillPoly(imgres, [warped_corners[:2, :].T], 0)
print(warped_corners)

res = warped_image + imgres

#using the median blur function to reduce the line-like noise

blur = cv.medianBlur(res, ksize = 5)
cv.imshow('line features', line_features)
cv.imshow('imperfect stitch', res)
cv.imshow('Stitch', blur)
cv.waitKey(0)
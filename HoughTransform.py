# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:31:19 2019

@author: Mohammadreza
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = plt.imread('ImageFile', format=None)
img = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

plt.imshow(img,cmap='gray')
plt.figure()

#edges = cv2.Canny(img,100,200)

#plt.imshow(edges)
#plt.figure()
res, img_binary = cv2.threshold(img,127,1,cv2.THRESH_BINARY)

#print(img_binary.tolist())
img1 = img_binary - 1
#


print(img1.tolist())
#print(img1.tolist())
#print(img1)

#img_binary = np.array([[0,0,0,0,0],[0,1,1,1,0],[0,0,0,0,0] ])
#print(img_binary)
#img1 = img_binary -1

#plt.imshow(img1,cmap='gray')
#plt.figure()

#print

height = img_binary.shape[0]
width = img_binary.shape[1]




#plt.imshow(img_binary)
#plt.figure()

#empty = np.zeros((height,width))

#print(empty)

thetas = np.deg2rad(np.arange(-90.0, 90.0))
diag_len = int(np.ceil(np.sqrt(width * width + height * height)))
p = np.linspace(-diag_len, diag_len, diag_len * 2.0)

#print(diag_len)

kernel = np.zeros((2 * diag_len, len(thetas)),dtype='int32')


for row in range(width):
    for col in range(height):
        
        if img1[col][row] == 255:
            #print('0')
            count = 0
            
            for theta in thetas:
                p0 = int(round(row * np.cos(theta) + col * np.sin(theta)) + diag_len)
                #print(p0)
                kernel[p0][count] += 1
                count = count + 1
                #print(kernel.tolist())
threshold = 115            
for row1 in range(kernel.shape[0]):
    for col1 in range(kernel.shape[1]):
        #print(kernel.shape[0])
        if kernel[row1][col1] > threshold:
            #print('o')
            if np.sin(thetas[col1]) != 0:
                m = np.divide(-1*(np.cos(thetas[col1])),np.sin(thetas[col1]))
                b = np.divide(p[row1],np.sin(thetas[col1]))

                x0 = 0
                y0 = int(b)
                #print('l')
                x1 = img1.shape[1]
                y1 = int(m * img1.shape[1] + b)
                
                cv2.line(img,(x0,y0),(x1,y1),(0,0,0))
                #print('j')
                #print((y0,y1))
plt.imshow(img,cmap='gray')
plt.figure()
cv2.waitKey(0)
cv2.destroyAllWindows()

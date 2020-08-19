# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 20:26:57 2020

@author: sooyi
"""
import cv2
import numpy as np
import math
ref_point = []

def norm(x):
    x=(x-x.min())/(0.001*x.max()-x.min())
    return x

def selection(ref_point,rows,cols):
    crow,ccol = round(rows/2) ,round(cols/2)
    mask= np.zeros((rows,cols))
    mask[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]= 1
    fft_ui1=fft_ui*mask
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(fft_ui1)
    My,Mx=maxLoc
    ffft_ui=np.zeros((rows,cols))
    ffft_ui=ffft_ui+0j
    ffft_ui[crow-40:crow+40, ccol-40:ccol+40] = dft_complex[Mx-40:Mx+40,My-40:My+40]
    return ffft_ui

def shape_selection(event, x, y, flags, param): 
    # grab references to the global variables 
    global ref_point,ref_point1, crop 
    if event == cv2.EVENT_LBUTTONDOWN: 
        ref_point = [(x, y)] 
    elif event == cv2.EVENT_LBUTTONUP: 
        ref_point.append((x, y)) 
        cv2.rectangle(image, ref_point[0], ref_point[1], (255, 0, 0), 1) 
        cv2.imshow("image", image)
        
def reconstruct(ffft_ui,dist,lamta):
    global dy,dx,dfx,dfy,yd,xd
    k = 2*math.pi/lamta
    m=np.zeros((yd,xd))
    n=np.zeros((yd,xd))
    [n,m]=np.mgrid[-yd/2:yd/2,-xd/2:xd/2]
    t_f=np.exp(1j*k*dist*np.sqrt(1-np.power((lamta*dfx*m),2)-np.power((lamta*dfy*n),2)))
    eff_frq_d=np.multiply(ffft_ui,t_f)
    ifft_ui=np.fft.ifftshift(eff_frq_d)
    ifft_ui=np.fft.ifft2(ifft_ui)
    return ifft_ui
        
img=cv2.imread(r'C:\Users\sooyi\Downloads\20150626\untitled1.bmp',0)
yd, xd = img.shape
yd=int(yd)
xd=int(xd)
crow,ccol = round(yd/2) ,round(xd/2)
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
dft_complex= dft_shift[:,:,0]+1j*dft_shift[:,:,1]
fft_ui = np.power(abs(dft_complex),2)
fft_norm = norm(fft_ui)
image = fft_norm
clone = image.copy() 
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 1000,700) 
cv2.setMouseCallback("image", shape_selection)

# keep looping until the 'q' key is pressed 
while True: 
    # display the image and wait for a keypress 
    cv2.imshow("image", image) 
    key = cv2.waitKey(1) & 0xFF
  
    # press 'r' to reset the window 
    if key == ord("r"): 
        image = clone.copy() 
  
    # if the 'c' key is pressed, break from the loop 
    elif key == ord("c"):
        break
    
if len(ref_point) == 2: 
    ffft_ui=selection(ref_point,yd,xd)
    ffft_norm = np.power(abs(ffft_ui),2)
    ffft_norm=norm(ffft_norm)
    cv2.imshow("image", ffft_norm)
    cv2.waitKey(0)

cv2.destroyAllWindows()

dist=0e-3
lamta=632e-9
dy=4.65e-6
dx=4.65e-6
dfx=1/(xd*dx)
dfy=1/(yd*dy)
ifft_ui=reconstruct(ffft_ui,dist,lamta)
phase=np.angle(ifft_ui)
phase=norm(phase)
cv2.imshow("reconstruct",phase)
cv2.waitKey(0)
cv2.destroyAllWindows()
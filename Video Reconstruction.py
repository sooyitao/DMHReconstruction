# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 20:54:52 2020

@author: sooyi
"""

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def shape_selection(event, x, y, flags, param): 
    # grab references to the global variables 
    global ref_point,ref_point1, crop 
    if event == cv2.EVENT_LBUTTONDOWN: 
        ref_point = [(x, y)] 
    elif event == cv2.EVENT_LBUTTONUP: 
        ref_point.append((x, y)) 
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2) 
        cv2.imshow("image", image)

def shape_selection1(event, x, y, flags, param): 
    # grab references to the global variables 
    global ref_point,ref_point1, crop 
    if event == cv2.EVENT_LBUTTONDOWN: 
        ref_point1 = [(x, y)] 
    elif event == cv2.EVENT_LBUTTONUP: 
        ref_point1.append((x, y)) 
        cv2.rectangle(image, ref_point1[0], ref_point1[1], (0, 255, 0), 2) 
        cv2.imshow("image", image)

def selection(ref_point,rows,cols):
    crow,ccol = round(rows/2) ,round(cols/2)
    mask= np.zeros((rows,cols))
    mask[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]= 1
    fft_ui1=fft_ui*mask
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(fft_ui1)
    My,Mx=maxLoc
    ffft_ui=np.zeros((rows,cols))
    ffft_ui=ffft_ui+0j
    ffft_ui[crow-30:crow+30, ccol-30:ccol+30] = dft_complex[Mx-30:Mx+30,My-30:My+30]
    return ffft_ui

def re_size(img,resize):
    global yd,xd
    y1=round((resize-yd)/2)
    x1=round((resize-xd)/2)
    x = np.pad(img, pad_width=[(y1,y1),(x1,x1)], mode='constant', constant_values=0)
    return x

def norm(x):
    x=(x-x.min())/(x.max()-x.min())
    return x

def reconstruct(ffft_ui,dist,lamta):
    global dy,dx,dfx,dfy,resize
    k = 2*math.pi/lamta
    m=np.zeros((resize,resize))
    n=np.zeros((resize,resize))
    [n,m]=np.mgrid[-resize/2:resize/2,-resize/2:resize/2]
    t_f=np.exp(1j*k*dist*np.sqrt(1-np.power((lamta*dfx*m),2)-np.power((lamta*dfy*n),2)))
    eff_frq_d=np.multiply(ffft_ui,t_f)
    ifft_ui=np.fft.ifftshift(eff_frq_d)
    ifft_ui=np.fft.ifft2(ifft_ui)
    return ifft_ui

def untilt(Uobj):
    global yd,xd
    [x,y]=np.mgrid[1-xd/2:xd/2,1-yd/2:yd/2]
    out_xd=920
    out_yd=690
    x_line=x[0,round((xd-out_xd)/2):round((xd+out_xd)/2)]
    y_line=y[round((yd-out_yd)/2):round((yd+out_yd)/2),0]
    Uobj=Uobj[round((yd-out_yd)/2):round((yd+out_yd)/2),round((xd-out_xd)/2):round((xd+out_xd)/2)]
    hor_line=Uobj[round(out_yd/2-300),:]
    ver_line=Uobj[:,round(out_xd/2-300)]
    hor_poly= np.polyfit(x_line, hor_line, 1);
    ver_poly= np.polyfit(y_line, ver_line, 1);
    xx=np.tile(np.polyval(hor_poly,x[round((xd-out_xd)/2):round((xd+out_xd)/2),0]),[out_yd,1])
    yy=np.tile(np.polyval(ver_poly,y[0,round((yd-out_yd)/2):round((yd+out_yd)/2)]),[out_xd,1])
    yy=np.rot90(yy,k=1,axes=(0,1))
    phase_factor=np.exp(-1j*(xx+yy))
    ifft_ui=Uobj*phase_factor
    return ifft_ui

vidcap = cv2.VideoCapture(r'C:\Users\sooyi\OneDrive\Desktop\Media1.mp4')
def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(r"C:\Users\sooyi\Downloads\data\frames"+str(count)+".BMP", image)
    return hasFrames
sec = 0
frameRate = 0.5 #//it will capture image in each 0.5 second
count=1
success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
    
img=cv2.imread(r'C:\Users\sooyi\Downloads\data\frames1.BMP',0)
yd, xd = img.shape
resize=1200
img=re_size(img,resize)
yd,xd=img.shape
crow,ccol = round(yd/2) ,round(xd/2)
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
dft_complex= dft_shift[:,:,0]+1j*dft_shift[:,:,1]
fft_ui = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
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
    
cv2.namedWindow("image") 
cv2.setMouseCallback("image", shape_selection1) 
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
    
i=1
while i < count:
    img=cv2.imread(r"C:\Users\sooyi\Downloads\data\frames"+str(i)+".BMP",0)
    yd, xd = img.shape
    resize=1200
    img=re_size(img,resize)
    yd,xd=img.shape
    crow,ccol = round(yd/2) ,round(xd/2)
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    dft_complex= dft_shift[:,:,0]+1j*dft_shift[:,:,1]
    fft_ui = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    fft_norm = norm(fft_ui)
    image = fft_norm
    ffft_ui=selection(ref_point,yd,xd)
    ffft_norm=norm(ffft_ui)
    ffft_ui1=selection(ref_point1,yd,xd)
    ffft_norm1=norm(ffft_ui1)
    
    dist=-200e-3
    lamta=632e-9
    dist1=-200e-3
    lamta1=637e-9
    dy=4.65e-6
    dx=4.65e-6
    dfx=1/(xd*dx)
    dfy=1/(yd*dy)
    ifft_ui=reconstruct(ffft_ui,dist,lamta)
    ifft_ui=untilt(ifft_ui)
    ifft_ui1=reconstruct(ffft_ui1,dist1,lamta1)
    ifft_ui1=untilt(ifft_ui1)
    ifft_ui2=np.divide(ifft_ui,ifft_ui1)
    phase=np.angle(ifft_ui)
    phase=norm(phase)
    phase1=np.angle(ifft_ui1)
    phase1=norm(phase1)
    phase2=np.angle(ifft_ui2)
    phase2n=norm(phase2)
    # cv2.imshow("first",phase)
    # cv2.imshow("second",phase1)
    cv2.imshow("reconstructed",phase2n)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    i=i+1
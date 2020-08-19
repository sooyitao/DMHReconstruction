# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 10:56:03 2020

@author: sooyi
"""

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog

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

root = tk.Tk()
root.withdraw()
root.wm_attributes("-topmost", 1)
file_path = filedialog.askopenfilename()
img=cv2.imread(file_path,0)
yd, xd = img.shape
resize=1200
img=re_size(img,resize)
yd,xd=img.shape
crow,ccol = round(yd/2) ,round(xd/2)
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
dft_complex= dft_shift[:,:,0]+1j*dft_shift[:,:,1]
fft_ui = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

mask=np.ones((yd,xd))
mask[crow-20:crow+20,ccol-20:ccol+20]=0
fft_ui=fft_ui*mask

q1=[(crow+10,10),(yd-10,ccol-10)]
q2=[(10,10),(crow-10,ccol-10)]
q3=[(10,ccol+10),(crow-10,yd-10)]
q4=[(crow+10,ccol+10),(xd-10,yd-10)]

ffft_ui=selection(q1,yd,xd)
ffft_ui1=selection(q2,yd,xd)


dist=-201e-3
lamta=637e-9
dist1=-201e-3
lamta1=632e-9
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
cv2.imshow("first",phase)
cv2.imshow("second",phase1)
cv2.imshow("reconstructed",phase2n)
cv2.waitKey(0)
cv2.destroyAllWindows()
# T=np.where(phase2<2)
# phase2[T]=phase2[T]+2*math.pi


beatwave=(lamta*lamta1)/(lamta1-lamta)
h=(beatwave/(4*math.pi))*phase2
M=cv2.mean(h[1:200,1:200])
h=h-M[0]
plt.plot(h[300,:])

x = range(920)
y = range(690)
hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
X, Y = np.meshgrid(x, y) 
surf=ha.plot_surface(X, Y, h,cmap='bwr', linewidth=0, antialiased=False)
plt.show()
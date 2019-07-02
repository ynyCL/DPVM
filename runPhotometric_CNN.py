import tensorflow as tf
import numpy as np
import cv2
import os
import shutil
import json
from PIL import Image
from PIL import ImageDraw
import time
import sys

import argparse
parser = argparse.ArgumentParser(description='Photometric CNN visibility metric')
parser.add_argument('--refpath', type=str, default='data/roofs-01_l1_bn_ref.png', help='path for reference image')
parser.add_argument('--dstpath', type=str, default='data/roofs-01_l1_bn_dst.png', help='path for distorted image')
parser.add_argument('--display', type=str, default='RGBbt709', help='display model type,  RGBbt709 implemented')
parser.add_argument('--peaklum', type=float, default=110, help='peak luminance of display')
parser.add_argument('--ppd', type=float, default=60., help='pixel per degree')
args = parser.parse_args()

print('=============================Photometric CNN==============================')
patchMergingMethod = 'average' 

# Parameters
batch_size = 1024
patchSize = 48
stride = 42

windowStep = patchSize - stride;
if(windowStep <= 0):
    windowStep = patchSize;

    

    
#Loading net    
ckpt = tf.train.get_checkpoint_state("DPVM_checkpoint")
print(ckpt);
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
print(saver);

pred = tf.get_collection("pred")[0]
x = tf.get_collection("x")[0]
x_ref = tf.get_collection("x_ref")[0]
keep_prob = tf.get_collection("keep_prob")[0]
pix_per_deg_tf = tf.get_collection("ppd")[0]

# Launch the graph
# with tf.Session() as sess:
sess = tf.Session()
saver.restore(sess, ckpt.model_checkpoint_path)


#Read Images
ppd = args.ppd
refImage = cv2.imread(args.refpath)
dstImage = cv2.imread(args.dstpath)

#Change images according to conditions
refImage = Image.open(args.refpath);
refImage = refImage.convert("RGB");
dstImage = Image.open(args.dstpath);
dstImage = dstImage.convert("RGB");

baseppd = 60.;
ratio = baseppd/ppd
print('ppd = ', ppd, 'use adjusted image ratio', ratio)
refImage = refImage.resize((int(refImage.size[0]*ratio), int(refImage.size[1]*ratio)), Image.ANTIALIAS)
dstImage = dstImage.resize((int(dstImage.size[0]*ratio), int(dstImage.size[1]*ratio)), Image.ANTIALIAS)


refImage = np.array(refImage)
dstImage = np.array(dstImage)

#value check:
if np.max(refImage) < 1.5:
    print('Warning! Image luminance too small. Please check whether the input image value is correct')

if args.display == 'RGBbt709': 
    from display_model import RGBbt_709_np
    refImage = RGBbt_709_np(refImage, args.peaklum)
    dstImage = RGBbt_709_np(dstImage, args.peaklum)
else:
    raise NotImplementedError('Display type not implemented!')


origWidth, origHeight, channels = refImage.shape;

tmpRefConcat = np.concatenate((np.flip(refImage, 0), refImage, np.flip(refImage, 0)),0);
refImage = np.concatenate((np.flip(tmpRefConcat, 1), tmpRefConcat, np.flip(tmpRefConcat, 1)),1)

tmpDstConcat = np.concatenate((np.flip(dstImage, 0), dstImage, np.flip(dstImage, 0)),0);
dstImage = np.concatenate((np.flip(tmpDstConcat, 1), tmpDstConcat, np.flip(tmpDstConcat, 1)),1)

refImage = refImage[origWidth-patchSize:2*origWidth+patchSize, origHeight-patchSize:2*origHeight+patchSize,:]
dstImage = dstImage[origWidth-patchSize:2*origWidth+patchSize, origHeight-patchSize:2*origHeight+patchSize,:]

width, height, channels = refImage.shape;

overlapNumber = int(patchSize / windowStep);
medianMaskImage = np.ones((width, height, overlapNumber*overlapNumber))
print(medianMaskImage.shape)

patchCounter = 0;



refRecords = [];
dstRecords = [];
xIndices = [];
yIndices = [];

luminRecords = [];
ppdRecords = [];

#Patches preparation
for i in range(0,width-patchSize,windowStep):
    for j in range(0,height-patchSize,windowStep):
        box = (j, i, j+patchSize, i+patchSize)
                    
        refPatch = refImage[i:i+patchSize,j:j+patchSize]                
        dstPatch = dstImage[i:i+patchSize,j:j+patchSize]              
        
        refRecords.append(refPatch)
        dstRecords.append(dstPatch)
        
        xIndices.append(i)
        yIndices.append(j)     

        #luminRecords.append(lumin)
        ppdRecords.append(ppd)
        
    
refTuple = tuple(refRecords)
dstTuple = tuple(dstRecords)  

ppdRecords = np.array(ppdRecords)

ppdRecords = np.reshape(ppdRecords, [ppdRecords.shape[0],])

times = len(refRecords) // batch_size
mod = len(refRecords) % batch_size
for i in range(times + 1):
    start = i*batch_size
    end = (i+1)*batch_size
    r = batch_size
    if(i == times):
        end = i*batch_size + mod
        r = mod
    
    if(len(dstRecords[start:end]) == 0 ):
        continue
    
    predict = sess.run(pred, feed_dict={x: tuple(dstRecords[start:end]) , x_ref:tuple(refRecords[start:end]), keep_prob: 1., pix_per_deg_tf:ppdRecords[start:end]  })

    for index in range(0, r):
        predictedPatch = cv2.resize(predict[index,:,:,:], (patchSize,patchSize));
        
        xx = xIndices[index+ i* batch_size]
        yy = yIndices[index+ i* batch_size]   
        
        layerIndex = int((xx%patchSize)/windowStep)*overlapNumber + int((yy%patchSize)/windowStep)
        medianMaskImage[xx:xx+patchSize, yy:yy+patchSize, layerIndex] = predictedPatch

if(patchMergingMethod == 'average'):
    maskImage = np.mean(medianMaskImage, axis=2);
if(patchMergingMethod == 'median'):
    maskImage = np.median(medianMaskImage, axis=2);
if(patchMergingMethod == 'max'):
    maskImage = np.max(medianMaskImage, axis=2);
if(patchMergingMethod == 'percentile'):
    maskImage = np.percentile(medianMaskImage, 95, axis=2);
    
print(maskImage.shape)   
maskImage = maskImage[patchSize:origWidth+patchSize, patchSize:origHeight+patchSize]
cv2.imwrite('prediction_np.png', maskImage)
    
   

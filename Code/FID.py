import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score
import cv2
import matplotlib.pyplot as plt
from numpy import asarray
from numpy import expand_dims
from numpy import log
from numpy import mean,std
from numpy import exp
from PIL import Image

#################################################################################
def calculate_fid(dset, gan):
	# calculate activations
	model = load_model('DenseNet_Model_Path')
	act1 = model.predict(dset)
	act2 = model.predict(gan)
	

	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid



dset = []
str = 'Dataset_Images_Path'
for fname in os.listdir(str):
	if fname == '.DS_Store':
		continue
	img = Image.open(str + fname)
	img = np.asarray(img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = expand_dims(img , axis=2)
	dset.append(img)
dset = np.asarray(dset)
print(np.shape(dset))

gan = []
str = 'Gan_Generated_Images_Path'
for fname in os.listdir(str):
	if fname == '.DS_Store':
		continue
	img = Image.open(str + fname)
	img = np.asarray(img)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = expand_dims(img , axis=2)
	gan.append(img)
gan = np.asarray(gan)
print(np.shape(gan))


fid_score = calculate_fid(dset , gan)
print('FID SCORE:')
print(fid_score)



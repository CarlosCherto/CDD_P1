from sklearn.cluster import KMeans
import numpy as np
from scipy import stats as st
from mnist import *
import pickle
from random import randint
import os
import matplotlib.pyplot as plt
import cv2

train_images = read_idx('train-images.idx3-ubyte')
train_labels = read_idx('train-labels.idx1-ubyte')

image_quantity = 50000
cluster_quantity = 1000

exec('kmeans = pickle.load(open("train_' + str(cluster_quantity) + 'clust_' + str(image_quantity) + 'imgs.p", "rb"))')

def cluster_dict():
	equivalencia = {}

	for cluster in range(0,cluster_quantity):
		lista_reais = []
		for index in range(0,image_quantity):
			if kmeans.labels_[index] == cluster:
				lista_reais.append(train_labels[index])
		
		equivalencia[str(cluster)] = st.mode(lista_reais, axis=None)[0][0]

	return (equivalencia)

def run_stats():
	
	statistics = {
					'Correct' : 0,
					'Incorrect' : 0
	}

	tries = 1000

	for n in range(0,tries):
		
		pred = randint(image_quantity,len(train_images)-1)

		if equivalencia[str(kmeans.predict([np.concatenate((train_images[pred]))])[0])] == train_labels[pred]:
			statistics['Correct'] += 1
		else:
			statistics['Incorrect'] += 1

	print('% de acertos: ' + str(statistics['Correct']*100/tries))
	print('% de erros: ' + str(statistics['Incorrect']*100/tries))

equivalencia = cluster_dict()

run_stats()

def focus(img):

	column_means = np.array([int(round(np.mean(img[x]))) for x in range(0,len(img))])

	ymin = 0
	ymax = len(img) - 1
	
	while column_means[ymin] < list(set(column_means))[1]:
		ymin += 1

	while column_means[ymax] < list(set(column_means))[1]:
		ymax -= 1

	ymin -= int(round((ymax - ymin)/7))
	ymax += int(round((ymax - ymin)/7))

	line_means = [int(round(np.mean(img[ymin:ymax,x]))) for x in range(0,len(img[0]))]

	xmin = 0
	xmax = len(line_means) - 1

	while line_means[xmin] <= list(set(line_means))[1]:
		xmin += 1

	while line_means[xmax] <= list(set(line_means))[1]:
		xmax -= 1

	if ((ymax - ymin) < (xmax - xmin)):
		if ((xmax - xmin) - (ymax - ymin))%2 != 0:
			diff = (((xmax - xmin) - (ymax - ymin)) + 1)/2
			ymax += diff
			ymin -= diff
		else:
			diff = ((xmax - xmin) - (ymax - ymin))/2
			ymax += diff
			ymin -= diff
	if ((ymax - ymin) > (xmax - xmin)):
		if ((ymax - ymin) - (xmax - xmin))%2 != 0:
			diff = (((ymax - ymin) - (xmax - xmin)) + 1)/2
			xmax += (diff - 1)
			xmin -= diff
		else:
			diff = ((ymax - ymin) - (xmax - xmin))/2
			xmax += diff
			xmin -= diff

	if (ymax - ymin)%28 != 0:
		diff = ((ymax - ymin)%28)
		if diff%2 != 0:
			diff += 1
		diff = diff/2
		ymax += diff
		ymin -= diff
		xmax += diff
		xmin -= diff

	ymax = int(ymax)
	ymin = int(ymin)
	xmax = int(xmax)
	xmin = int(xmin)

	img = img[ymin:ymax,xmin:xmax]

	new_img = []

	length = len(img)/28
	
	for y in range(0,28):
		line = []
		for x in range(0,28):
			line.append(int(round(np.mean(img[int(round(length*y)):int(round(length*(y + 1))),int(round(length*x)):int(round(length*(x + 1)))]))))
		new_img.append(line)

	return (new_img)

def guesser(img):
	return((equivalencia[str(kmeans.predict([np.concatenate(focus(img))])[0])]))


cap = cv2.VideoCapture(0)

counter = 0

number = []

while(1):

	# Take each frame
	_, frame = cap.read()
	
	# Convert BGR to HSV
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	
	# define range of blue color in HSV
	lower_blue = np.array([110, 100, 20])
	upper_blue = np.array([160, 255, 255])

	# Threshold the HSV image to get only blue colors
	mask = cv2.inRange(hsv, lower_blue, upper_blue)

	try:
		if len(number) <= 20:
			number.append(guesser(mask))
		else:
			number = number[1:]
			number.append(guesser(mask))
			print('Number found: ' + str(st.mode(number)[0][0]))
	except ValueError:
		number = []
		print('Looking for number...')
	except IndexError:
		number = []
		print('Looking for number...')
	
	counter += 1
	
	cv2.imshow('mask',mask)
	cv2.imshow('frame',frame)
	k = cv2.waitKey(5) & 0xFF
	if k == 27:
		break

cv2.destroyAllWindows()


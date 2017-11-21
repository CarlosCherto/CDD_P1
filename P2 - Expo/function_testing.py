import numpy as np
import cv2
from sklearn.cluster import KMeans
from scipy import stats as st
from mnist import *
import matplotlib.pyplot as plt

img = cv2.imread('test_img_paint_num_4.jpg')

img = 255 - img[:,:,2]

def focus(img):

	column_means = np.array([int(round(np.mean(img[x]))) for x in range(0,len(img))])

	ymin = 0
	ymax = len(img) - 1

	print(list(set(column_means)))

	while column_means[ymin] < list(set(column_means))[1]:
		ymin += 1

	while column_means[ymax] < list(set(column_means))[1]:
		ymax -= 1

	ymin -= int(round((ymax - ymin)/7))
	ymax += int(round((ymax - ymin)/7))

	line_means = [int(round(np.mean(img[ymin:ymax,x]))) for x in range(0,len(img[0]))] # 'x' é a funcao abaixo...
	
	xmin = 0
	xmax = len(line_means) - 1

	while line_means[xmin] <= list(set(line_means))[3]:
		xmin += 1

	while line_means[xmax] <= list(set(line_means))[3]:
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

	# int(round(np.mean(img[ymin:ymax,0])))

# print(focus(img))
# plt.imshow(focus(img))
# plt.show()

focus(img)

# print(img[100:110,0:10])

# print(int(round(np.mean(img[10:20,0]))))
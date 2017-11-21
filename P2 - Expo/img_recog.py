import numpy as np
import cv2
from math import *
from sklearn.cluster import KMeans
from scipy import stats as st
from mnist import *

# train_images = read_idx('train-images.idx3-ubyte')
# train_labels = read_idx('train-labels.idx1-ubyte')

# index_lim = 5000
# X = np.array([np.concatenate((train_images[x])) for x in range(0,index_lim)])

# clustquant = 500
# kmeans = KMeans(n_clusters=clustquant, random_state=0).fit(X)

# equivalencia = {}

# for cluster in range(0,clustquant):
#     lista_reais = []
#     for index in range(0,index_lim):
#         if kmeans.labels_[index] == cluster:
#             lista_reais.append(train_labels[index])
    
#     equivalencia[str(cluster)] = st.mode(lista_reais, axis=None)[0][0]


# def guesser(img):

#     #Converte pra HSV
#     img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2])

#     new_mean = np.mean(img)

#     ymin = 0
#     ymax = len(img)
#     xmin = 0
#     xmax = len(img[0])

#     while np.mean(img[ymin:ymax,xmin + 1:xmax]) < new_mean:
#         xmin = xmin + 1
#         new_mean = np.mean(img[ymin:ymax,xmin:xmax])

#     while np.mean(img[ymin:ymax,xmin:xmax - 1]) < new_mean:
#         xmax = xmax - 1
#         new_mean = np.mean(img[ymin:ymax,xmin:xmax])

#     while np.mean(img[ymin + 1:ymax,xmin:xmax]) < new_mean:
#         ymin = ymin + 1
#         new_mean = np.mean(img[ymin:ymax,xmin:xmax])

#     while np.mean(img[ymin:ymax - 1,xmin:xmax]) < new_mean:
#         ymax = ymax - 1
#         new_mean = np.mean(img[ymin:ymax,xmin:xmax])

#     if (ymax - ymin)%28  != 0:
#         while ((ymax - ymin)%28)%2 != 0:
#             ymax += 1
#         diff = ((ymax - ymin)%28)/2
#         ymax += diff
#         ymin -= diff

#     if (xmax - xmin)%28  != 0:
#         while ((xmax - xmin)%28)%2 != 0:
#             xmax += 1
#         diff = ((xmax - xmin)%28)/2
#         xmax += diff
#         xmin -= diff

#     img = img[int(ymin):int(ymax),int(xmin):int(xmax)]

#     img = 255 - img

#     xpxwidth = len(img[0])/28

#     ypxwidth = len(img)/28

#     new_img = []

#     for y in range(0,28):
#         line = []
#         for x in range(0,28):
#             line.append(int(round(np.mean(img[int(ypxwidth*y):int(ypxwidth*(y + 1)),int(xpxwidth*x):int(xpxwidth*(x + 1))]))))
#         new_img.append(line)

#     new_img = (np.array(new_img))
#     plt.imshow(new_img)
#     plt.show()
#     new_img = np.concatenate(new_img)
#     print(print(equivalencia[str(kmeans.predict([new_img])[0])]))

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
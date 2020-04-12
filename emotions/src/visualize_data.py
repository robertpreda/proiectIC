import pandas as pd
import numpy as np
import cv2

import matplotlib.pyplot as plt

csv_file_path = '../data/fer2013.csv'
faces_dataframe = pd.read_csv(csv_file_path)
# print(faces_dataframe.head())
data = faces_dataframe.iloc[0, 1].split(' ')
converted_data = []
for d in data:
    # print(d)
    # exit()
    temp = int(d)
    converted_data.append(d)
converted_data = np.array(converted_data).astype(np.uint8)
converted_data = converted_data.reshape((48,48))
# print(type(converted_data[0,0]))
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 480,480)
cv2.imshow("Image", converted_data)
cv2.waitKey()
# cv2.imwrite("output.png",converted_data)
# plt.imshow(converted_data)
# plt.show()
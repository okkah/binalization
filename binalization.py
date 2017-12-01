# -*- coding: utf-8 -*-

import numpy as np
import pylab as plt
from PIL import Image
from matplotlib.image import imread

#グレースケール化
img=Image.open('okkah.png')
gray=img.convert('L')
gray.save('okkah_gray.png')

#ヒストグラムを作成
img=imread('okkah_gray.png')
img=np.array(img)
input_shape = img.shape
print("input_shape =", input_shape)	#入力画像の大きさを出力
img=img.flatten()*255
plt.hist(img, bins=256, range=(0, 255))
plt.savefig('okkah_histgram.png')

histgram = np.zeros(256)
for i in img:
	histgram[int(i)] += 1

#大津の閾値判定法
max_t = max_val = 0

for t in range(0, 256):
	w1 = w2 = 0			#画素数初期化
	sum1 = sum2 = 0		#クラス別合計値初期化
	m1 = m2 = 0.0		#クラス別平均値初期化
	for i in range(0, t):
		w1 += histgram[i]
		sum1 += i*histgram[i]
	for j in range(t, 256):
		w2 += histgram[j]
		sum2 += j*histgram[j]
	if w1 == 0 or w2 == 0:	#0で割ることを禁止
		continue
	m1 = sum1/w1		#クラス別平均値を求める
	m2 = sum2/w2
	result = w1*w2*(m1-m2)*(m1-m2)	#結果を求める
	if max_val < result:
		max_val = result
		max_t = t
print ("max_t =", max_t)			#閾値を出力

#求めた閾値を基に画像の２値化を行う
output = np.copy(img)
output[output < max_t] = 0
output[output >= max_t] = 255

#画像の表示
output = output.reshape(input_shape)
print(output)

plt.clf()
plt.imshow(output, cmap='gray')
plt.show()

import cv2
import numpy as np

# Haar-like特徴分類器の読み込み
face_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_default.xml')

#
# 静画像に対する顔認識 
#

# イメージファイルの読み込み
img = cv2.imread('sample.jpg')

# 評価器に画像ファイルを通す
faces = face_cascade.detectMultiScale(img,minNeighbors=10)
for (x,y,w,h) in faces:
    #検知した顔を矩形で囲む
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3) # rectangle(対象,始点,終点,線の色(B,G,R),線の太さ)

# 顔を矩形に囲んだ画像を表示
cv2.imshow('image: face detected', img)

# 検出された顔部分を画像表示
cv2.imshow('image: face only', img[y:y+h, x:x+w])

cv2.waitKey(0)
cv2.destroyAllWindows()


#
# 動画像に対する顔認識
#

# カメラデバイスの番号を指定
cap = cv2.VideoCapture(0)

while True:
    # cv2.videocapture.read()は返り値として画像取得の成否と取得画像を返す
    flag, frame = cap.read()

    faces = face_cascade.detectMultiScale(frame,minNeighbors=10)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

    cv2.imshow('frame: face detected', frame)
    cv2.imshow('face only', frame[y:y+h, x:x+w])

    # escキーを入力して終了する
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()


#
# 静画像に対する顔の入れ変え（アルファチャンネルを考慮）
#

# 入れ替える画像
kora = cv2.imread("kora.png", cv2.IMREAD_UNCHANGED)
img = cv2.imread('sample.jpg')

# 評価器に画像ファイルを通す
faces = face_cascade.detectMultiScale(img,minNeighbors=10)

kora = cv2.resize(kora, (h, w))

width, height = kora.shape[:2] # コラ画像用のサイズを取得

mask_ = kora[:,:,3] # アルファチャンネルのみを抜き出す
# アルファチャンネルの値をＢＧＲに格納したndarrayを作る
mask = np.ones((width, height, 3))
for i in range(len(mask)):
    for j in range(len(mask[0])):
        mask[i][j] = np.ones(3) * mask_[i][j]
mask = mask / 255.0 # 0.0 ~ 1.0に変更
mask = mask.astype('uint8') # maskをimgと同一のuint8に変換する

kora = kora[:,:,:3] # アルファチャンネル以外を取り出す

img[y:y+height, x:x+width] *= 1 - mask # 透過率に応じて元の画像を暗くする
img[y:y+height, x:x+width] += mask * kora # koraに透過率をかけて加算する

cv2.imshow('image: face detected', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#
# 動画像に対する顔の入れ変え（アルファチャンネルを考慮）
#

cap = cv2.VideoCapture(0)
kora = cv2.imread("kora.png", cv2.IMREAD_UNCHANGED)

kora = cv2.resize(kora, (h, w))
width, height = kora.shape[:2]

mask_ = kora[:,:,3]
mask = np.ones((width, height, 3))
for i in range(len(mask)):
    for j in range(len(mask[0])):
        mask[i][j] = np.ones(3) * mask_[i][j]
mask = mask / 255.0
mask = mask.astype('uint8')

kora = kora[:,:,:3]

while True:
    flag, frame = cap.read()

    faces = face_cascade.detectMultiScale(frame,minNeighbors=10)

    for (x,y,w,h) in faces:
        # 顔の座標を検出
        pass
    frame[y:y+height, x:x+width] *= 1 - mask
    frame[y:y+height, x:x+width] += mask * kora
    cv2.imshow('Kora Gazo', frame)

    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

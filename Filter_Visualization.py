import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
# slim은 CNN을 만들 때 반복적으로 사용되는 변수 선언 등을 추상화할 수 있는 고수준 경량 API
from tensorflow.examples.tutorials.mnist import input_data
# 로컬 폴더에서 MNIST 데이터셋을 찾고, 데이터가 없다면 인터넷에서 MNIST 데이터셋을 가져와서 사용자의 로컬 저장소에 내려받는다

# 예제 6.2 input_data 함수를 호출해 MNIST 데이터세트를 내려받는 코드

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 예제 6.3 합성곱 신경망 구축하기

tf.reset_default_graph()
# 주피터 노트북 환경에서 텐서플로 그래프가 두 개씩 중복으로 생성되는 일을 막음
# 주피터 노트북은 매회 실행에도 컨텍스트가 유지되기 때문에 노트북을 사용한다면 그래프 충돌이 발생할 수 있다

x = tf.placeholder(tf.float32, [None, 784], name="x-in")
# placeholder(): CNN에 들어갈 이벽 벡터의 모양을 선언하고 전달
# 입력 변수 x는 실수 자료형에 , 28 x 28 크기
# 파라미터의 순서: 자료형, 모양(여기서는 2차원), 명칭
# None은 텐서플로 플레이스홀더에서 '어떤 크기도 들어올 수 있다'는 의미

true_y = tf.placeholder(tf.float32, [None, 10], name="y-in")
# 학습에 사용하기 위한 레이블이 들어감. 레이블의 갯수는 0~9까지 총 10개의 클라스이다.

keep_prob = tf.placeholder("float")
# CNN의 overfitting을 막기 위해 dropout을 조정하는 파라미터
# kepp_proba가 1.0이라면 행렬 연산 결과에 대해 100%를 보전하고, 0.2라면 행렬 연산 결과에 대해 20%를 남기고 나머지 결과는 모두 무시

x_image = tf.reshape(x,[-1,28,28,1])
# 입력 벡터 784개(28 x 28)를 처리. 28 x 28 x 1 크기의 행렬로 변환.
# 신경망에 입력된 MNIST 이미지의 데이터 개수는 정해지지 않았음(-1)

hidden_1 = slim.conv2d(x_image,5,[5,5])
# 첫번째 hidden layer. 28 x 28(이미지) 행렬에 5 x 5 크기의 필터 5개를 합성곱
# 이차원 이미지를 연산하기 위해 다섯 개 필터를 만들고, 크기가 1인 stride를 사용
# 행렬 합성곱 결과물로 28 x 28 크기의 2차원 행렬 다섯 장을 출력 (28 x 28 x 5)

pool_1 = slim.max_pool2d(hidden_1,[2,2])
# 2 x 2 크기의 필터로 맥스 풀링을 수행 -> 벡터의 크기는 반으로 줄어들음
# 따라서 (14 x 14 x 5)가 출력됨

hidden_2 = slim.conv2d(pool_1,5,[5,5])
# 위에 hidden layer와 마찬가지 방법으로 (14 x 14 x 5)가 출력됨

pool_2 = slim.max_pool2d(hidden_2,[2,2])
# (7 x 7 x 5)가 출력됨

hidden_3 = slim.conv2d(pool_2,20,[5,5])
# (7 x 7 x 20)가 출력됨

hidden_3 = slim.dropout(hidden_3,keep_prob)
# 출력에서 keep_prob 비율만큼 남기고 나머지 벡터를 연산 결과를 차단한다.

out_y = slim.fully_connected(
    slim.flatten(hidden_3), # hidden_3을 1차원 벡터로 평탄화
    10, # 라벨의 수(10)만큼 압축
    activation_fn=tf.nn.softmax) # 출력층의 활성 함수를 softmax로 설정

cross_entropy = -tf.reduce_sum(true_y*tf.log(out_y)) 
# Cross Entropt Error 기법을 사용
# 교차 엔트로피 에러는 엔트로피 이론에 근거해 모델의 출력과 실제 출력 사이의 오차를 계산

correct_prediction = tf.equal(tf.argmax(out_y,1), tf.argmax(true_y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# cross_entropy 결과가 최소로 줄어들게 최적화 함수를 조정
# 이때 조정한 최적화 값은 신경망의 가중치와 바이어스를 조절 -> 이 과정에서 오차를 후방으로 전달하는 back propagation이 사용

# 예제 6.7 직접 구현한 합성곱 신경망을 학습시키는 코드

batchSize = 50
# 배치 크기는 실행 한경이 좋다면 크게, 안좋다면 작게 하는게 좋음

sess = tf.Session()
# 세션 객체를 선언. 세션 객체는 자신만의 물리적 차원(CPU, GPU 등)을 점유

init = tf.global_variables_initializer()
# 텐서플로의 각종 글로벌 변수를 초기화하는 코드

sess.run(init)
# 그래프를 초기화

for i in range(1000):
    batch = mnist.train.next_batch(batchSize)
    sess.run(train_step,
            feed_dict={x:batch[0],true_y:batch[1], keep_prob:0.5})
    if i % 100 == 0 and i != 0:
        trainAccuracy = sess.run(accuracy,
                feed_dict={x:batch[0],
                    true_y:batch[1],
                    keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, trainAccuracy))

# 예제 6.8 테스트 데이터를 활용해 학습된 신경망의 정확도를 구하는 코드

testAccuracy = sess.run(accuracy,
        feed_dict={x: mnist.test.images,
            true_y: mnist.test.labels,
            keep_prob: 1.0})

print("test accuracy %g"%(testAccuracy))

#### 여기서부터 합성곱 신경망 시각화하기 #####

# 예제 6.9 테스트 데이터 0번을 시각화하는 코드

index = 0 
imageToUse = mnist.test.images[index]
imageLabel = mnist.test.labels[index]
print(imageToUse.shape) # 0번재 테스트 데이터의 텐서 모양을 출력
print(imageLabel) # 원 핫 인코딩을 한 0번째 인덱스
plt.imshow(np.reshape(imageToUse, [28,28]),
        interpolation="nearest", cmap="gray")

# 예제 6.10 합성곱 신경망이 0번째 손글씨 이미지를 예측하는 코드

image_in = np.reshape(imageToUse, [1, 784]) # CNN의 입력 모양에 맞게 평탄화 작업 수행
arg_max = tf.argmax(out_y, 1) # 신경망 모델의 출력 결과에서 가장 값이 큰 인덱스를 반환
predict = sess.run(arg_max, 
        feed_dict={x: image_in, keep_prob: 1.0})
print(predict) # 모델이 예측한 결과를 저장

# 예제 6.11 합성곱 신경망이 예측한 데이터 라벨과 실제 데이터 라벨을 비교하는 코드

print(imageLabel.argmax())
print(predict[0])
print(predict[0] == imageLabel.argmax())

# 예제 6.12 테스트 데이터 924번을 시각화하는 코드

index = 924
imageToUse = mnist.test.images[index]
imageLabel = mnist.test.labels[index]
print(imageToUse.shape)
print(imageLabel)
plt.imshow(np.reshape(imageToUse, [28,28]), interpolation="nearest", cmap="gray")

# 예제 6.11을 924번에 대해 실행한 결과

print(imageLabel.argmax())
print(predict[0])
print(predict[0] == imageLabel.argmax())

# 예제 6.13 합성곱 신경망이 테스트 데이터 924번을 어떻게 예측하는지 숫자별로 확률을 보여주는 코드

mat = sess.run(out_y, feed_dict={x: image_in, keep_prob: 1.0})[0] # 텐서플로 세션에 출력 레이어(out_y) 값을 그대로 받음
# feed_dict라는 함수는 앞에 있는 파라미터(out_y)에 인자를 넣어주는것 .
count = 0
for i in mat:
    print('[{}] {:.2%}'.format(count, i))
    count += 1

# 예제 6.14 이미지 하나가 특정 은닉층까지 통과한 결과물을 units 변수에 저장하고 호출하는 함수
# 여기서부터 다시 보기

def getActivations(layer, stimuli):
    units = sess.run(
        layer, #은닉층을 받을 파라미터 . 출력물은 stimuli 입력값이 layer 계층까지 통과된 결과물
        feed_dict={
            x: np.reshape(stimuli, [1,784], order='F'),
            # stimuli 변형. 이미지를 그래프의 입력 텐서 x의 규격 대로( 1 x 784 )로 변형
            # F - Fortran : 인덱스 순서를 사용하여 요소를 읽고 쓰는 것을 의미. 첫 번째 인덱스가 가장 빠르게 변경되고 마지막 인덱스가 가장 느리게 변경
            keep_prob:1.0}) # dropout 보전치는 1.0
    tf.shape(units) 
    plotNNFilter(units)

# 예제 6.15 은닉층 연산 결과를 시각화하는 코드

import math
def plotNNFilter(units): # 은닉층 텐서를 받아서 시각. 은닉층 텐서는 1 x (행렬 연산 가로) x (행렬 연산 세로) x (필터 수) 크기.
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    # 첫 번째 파라미터는 객체 아이디로, 한 줄에 여러 이미지를 붙여서 출력하기 위해서 1이라고 함
    # 두 번째 파라미터는 이미지 크기

    n_columns = 5
    n_rows = math.ceil(filters / n_columns) + 1 # 필터의 줄 수를 저장. 필터의 개수가 21개라면 n_columns , n_rows 둘 다 5개 필요
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")
        plt.show()

# 예제 6.16 첫 번째 은닉층을 시각화하는 코드

getActivations(hidden_1, imageToUse)

# 예제 6.17 합성곱 신경망의 두 번째 은닉층을 시각화하는 코드

getActivations(hidden_2, imageToUse)

# 예제 6.18 세 번째 은닉층을 시각화하는 코드

getActivations(hidden_3, imageToUse)

# 예제 6.18 MNIST 손글씨 테스트 데이터베이스 223번째 데이터를 불러오는 코드

imageToUse = mnist.test.images[223]

# 2개 이상의 데이터에 대한 필터 결과 출력

def getActivationsMulti(layer, stimulis):
    units = []
    for stim in stimulis:
        unit = sess.run(
            layer,
            feed_dict={
                x: np.reshape(stim, [1,784], order='F'),
                keep_prob:1.0})
        units.append(unit)
    units = np.concatenate(units, 3)
    plotNNFilter(units)

getActivationsMulti(hidden_1, (mnist.test.images[924], mnist.test.images[223]))

getActivationsMulti(hidden_2, (mnist.test.images[924], mnist.test.images[223]))

getActivationsMulti(hidden_3, (mnist.test.images[924], mnist.test.images[223]))
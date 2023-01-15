# 예제 7.1 합성곱 신경망을 제작하고 정확도를 출력하는 코드

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim
# slim은 CNN을 만들 때 반복적으로 사용되는 변수 선언 등을 추상화할 수 있는 고수준 경량 API
from tensorflow.examples.tutorials.mnist import input_data

# extracting MNIST Dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# construct CNN
# set models tf.reset_default_graph()
# placeholder란, 처음에 변수를 선언할 때 값을 바로 주는 것이 아니라, 나중에 값을 던져주는 '공간'을 만들어주는 것
x = tf.placeholder(tf.float32, [None, 784],name="x-in") # 28 x 28 x 1 MNIST 데이터를 평탄화(1차원 배열로 바꿔줌)
# 순서대로 데이터 타입, 입력 데이터의 형태, 해당 placeholder의 이름 설정
true_y = tf.placeholder(tf.float32, [None, 10],name="y-in")
# true_y에는 0 ~ 9까지의 라벨이 들어감
keep_prob = tf.placeholder(dtype=tf.float32)
# tf.placeholder("float") -> 예전 버전이어서 가능, 현재는 tf.placeholder(dtype=tf.float32) 이렇게 적어야 함
x_image = tf.reshape(x,[-1,28,28,1]) #평탄화한 데이터를 다시 28 x 28 x 1로 바꿔줌
# x_image = tf.reshape(x,[-1,784]) #평탄화한 데이터를 다시 28 x 28 x 1로 바꿔줌
# print(x_image.shape)
# print(x_image)

# layer 1
hidden_1 = slim.conv2d(x_image,5,[5,5])
# print(hidden_1.shape)
# print(hidden_1)
# exit()
pool_1 = slim.max_pool2d(hidden_1,[2,2])
# print(pool_1.shape)

# layer 2
hidden_2 = slim.conv2d(pool_1,5,[5,5])
# print(hidden_2.shape)
pool_2 = slim.max_pool2d(hidden_2,[2,2])
# print(pool_2.shape)

# layer 3
hidden_3 = slim.conv2d(pool_2,20,[5,5])
#print(hidden_3.shape)
hidden_3 = slim.dropout(hidden_3,keep_prob)
print(hidden_3.shape)
#print()

out_y = slim.fully_connected(slim.flatten(hidden_3), 10, activation_fn=tf.nn.softmax)

cross_entropy = -tf.reduce_sum(true_y*tf.log(out_y))
correct_prediction = tf.equal(tf.argmax(out_y,1), tf.argmax(true_y,1))
# tf.equal 함수는 인자로 받은 두 텐서의 원소가 같을 경우, True를 같지 않다면 False를 반환
# tf.argmax(a,0) , tf.argmax(a,1) -> 0이면 각 '열'에서 가장 큰 값을 찾아 해당 값의 인덱스 번호를 반환. 1이면 각 '행'에서 가장 큰 값을 찾아 해당 값의 인덱스 번호를 반환
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# tf.cast 함수는 부동소수점형에서 소수형으로 바꾸고 소수점을 버림
# ex) 1.89999 -> 0 , 2.09999 -> 1
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# learning
batchSize = 50
sess = tf.Session()
# 세션을 만들고 연산그래프 실행
init = tf.global_variables_initializer() # 변수 초기화
sess.run(init) # 변수 초기화 수행

for i in range(1000):
    batch = mnist.train.next_batch(batchSize) # batchsize만큼 데이터를 가져와 학습을 진행
    sess.run(train_step, feed_dict={x:batch[0],true_y:batch[1], keep_prob:0.5})
    # x에 batch[0]을 넣고, y에 batch[1]을 넣는다. keep_prob:0.5는 0.5만큼 학습하는데 사용
    if i % 100 == 0 and i != 0:
        trainAccuracy = sess.run(accuracy, feed_dict={x:batch[0], true_y:batch[1], keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, trainAccuracy))
        # print(f"out_y shpae:{out_y.shape}")

# print Accuracy
print('Accuracy: {:.2%}'.format(sess.run(accuracy, feed_dict={x: mnist.test.images, true_y: mnist.test.labels, keep_prob: 1.0})))

# 예제 7.2 필요 변수 1, 2를 불러오는 코드
# 필요한 변수들
# 1. 신경망의 계층별 활성화 함수
# 2. 신경망의 두 계층 사이를 잇는 가중치 벡터
# 3. 분류 결과값 = 해당 코드에서의 out_y
# 4. 신경망의 앞 계층 노드별 타당성 수치는 분류 결과값으로부터 두 계층 사이를 잇는 가중치 벡터를 가공해서 구할 수 있다
layers = [hidden_1, pool_1, hidden_2, pool_2, hidden_3]
for i in layers:
    print(i.shape)

print('*'*10,'layers','*'*10)
for layer in layers:
    print("###############layer:",layer)
# get_collection 메서드는 텐서플로 세션에 올라간 모델 정보를 가져온다. get_collection 명령어는 모델의 가중치와 바이어스를 모두 불러올 수 있다.
# get_collection은 현재 텐서플로가 세션에 올려놓은 그래프 중 특정 변숫값을 불러 오는 메서드
weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*weights.*')
# 가중치(weigths)와 바이어스(bias)는 scope 변수를 조정해서 필터링한다
print('*'*10,'weights','*'*10)
for weight in weights:
    print(f"{weight}:{weight.shape}")
biases = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*biases.*')
print('*'*10,'biases','*'*10)
for bias in biases:
    print(f"{bias}:{bias}")
# bias 또한 마찬가지

# 예제 6.14(중복) 계층별 활성화 함수 결과를 출력하는 코드
def getActivations(layer, image):
    units = sess.run(layer, feed_dict={x: np.reshape(image, [1,784], order='F'), keep_prob:1.0})
    # print(f"#####################units:{units.shape}")
    return units

# 예제 7.4 LRP를 수행할 이미지 하나를 부르는 과정

# show test image
idx = 4
imageToUse = mnist.test.images[idx]
plt.imshow(np.reshape(imageToUse,[28,28]), interpolation="nearest", cmap='gray')

# 예제 7.5 합성곱 신경망의 은닉 계층마다 활성화 함수를 구하는 코드
# layers = [hidden_1, pool_1, hidden_2, pool_2, hidden_3]
activations = []
for layer in layers:
    print(layer)
    activations.append(getActivations(layer, imageToUse))
# print('#'*15,'4번째 사진','#'*15)
# a=1
# for i in activations:
#     print(a)
#     print(i)
#     print("길이:",len(i))
#     a+=1
# print("activations[0]: ",activations[0])
# print('#'*15)
# print(len(activations))

# 예제 7.6 합성곱 신경망에 이미지를 입력하고 out_y 예측 결과를 구하는 코드
predict = sess.run(out_y, feed_dict={x: imageToUse.reshape([-1, 784]), keep_prob:1.0})
print(predict)
predict = sess.run(out_y, feed_dict={x: imageToUse.reshape([-1, 784]), keep_prob:1.0})[0]
idx = 0
for i in predict:
    print('[{}] {:.2%}'.format(idx, i))
    idx += 1
print(predict)
# 예제 7.7 학습된 합성곱 신경망으로부터 분류 가능성이 최대가 되는 카테고리를 구하는 코드

f_x = max(predict)
# print("f_x:",f_x)
# print(type(f_x))

#######합성곱 신경망에 LRP 적용하기

# 예제 7.8 완전 연결 신경망에서 역전파 기울기를 구하는 수도 코드
# get FC layer gradient
def getGradient(activation, weight, bias):
    print("#"*20,"getGradient","#"*20)
    print("activation: ",activation)
    # forward pass
    W = tf.maximum(0., weight)
    print("Weight : ",W)
    b = tf.maximum(0., bias)
    print("Bias : ", b)
    z = tf.matmul(activation, w) + b
    print("Z : ",z)
    # backward pass
    dX = tf.matmul(1/z, tf.transpose(W))
    print("dX : ",dX)
    return dX

#  예제 7.9 f_x로부터 바로 직전 은닉층의 타당성 전파 값을 구하는 코드

R4 = predict
print(f"R4 shape:\n{R4.shape}")
print(f"R4:\n{R4}")

# 예제 7.9 FC 연결에서 LRP를 수행하는 코드. 예제 7.8의 역전파 기울기를 구하는 코드에 타당성 변수(relevance)를 곱한다

def backprop_dense(activation, weight, bias, relevance):
    print("#"*20,"backprop_dense","#"*20)
    print("activation: ",activation)
    print("relevance: ",relevance)
    w = tf.maximum(0., weight)
    print("weight: ",w)
    print(f"weight shape: {weight.shape}")
    b = tf.maximum(0., bias)
    print("bias: ",b)
    print(f"bias shape: {bias.shape}")
    z = tf.matmul(activation, w) + b
    print("z: ",z)
    print(f"z shpae: {z.shape}")
    s = relevance / z
    print("s: ",s)
    print(f"s shape: {s.shape}")
    print(f"transpose w: {tf.transpose(w).shape}")
    c = tf.matmul(s, tf.transpose(w))
    print("c: ",c)
    print(f"c shape: {c.shape}")
    return activation * c

# 예제 7.10 예제 7.9에서 만든 LRP 공식으로 𝑅3를 구하는 코드

# layers = [hidden_1, pool_1, hidden_2, pool_2, hidden_3]
# (1, 28, 28, 5) (1, 14, 14, 5) (1, 7, 7, 20)
# activation, weights, biases
a = activations.pop()
# print(f"a type:\n{type(a)}")
# print(f"a len:\n{a.shape}")
# print(f"a:\n{a}")
w = weights.pop()
# print(f"w type:\n{type(w)}")
# print(f"w len:\n{w.shape}")
# print(f"w:\n{w}")
b = biases.pop()
# print(f"b type:\n{type(b)}")
# print(f"b:\n{b}")

# print(f"a.shape: {a.shape}")
# print(f"w.shape: {w.shape}")

R3 = backprop_dense(a.reshape(1,980), w, b, R4)
print(R3)
print(f"R3.shape: {R3.shape}")
exit()


# 예제 7.11-(1) 언풀링 연산에서 LRP를 구하는 코드
from tensorflow.python.ops import gen_nn_ops

def backprop_pooling(activation, relevance):
    # kernel size, strides
    # if z is zero
    ksize = strides = [1, 2, 2, 1]
    z = tf.nn.max_pool(activation, ksize, strides, padding='SAME') + 1e-10
    s = relevance / z
    # input, argmax, argmax_mask
    c = gen_nn_ops._max_pool_grad(activation, z, s, ksize, strides, padding='SAME')
    print(f"c;\n{c}")
    return activation * c

# 예제 7.11-(2) 역합성곱 연산에서 LRP를 구하는 코드

def backprop_conv(activation, weight, bias, relevance):
    strides = [1, 1, 1, 1]
    w = tf.maximum(0., weight)
    b = tf.maximum(0., bias)
    z = tf.nn.conv2d(activation, w, strides, padding='SAME')
    z = tf.nn.bias_add(z, b)
    print(f"z:\n{z}")
    s = relevance / z
    c = tf.nn.conv2d_backprop_input(tf.shape(activation), w, s, strides, padding='SAME')
    return activation * c

# 예제 7.12 𝑅3 벡터로부터 역합성곱과 언풀링 연산을 수행하고 𝑅2 벡터를 구하는 코드

# layers = [hidden_1, pool_1, hidden_2, pool_2]
# (1, 28, 28, 5)(1, 14, 14, 5)(1, 7, 7, 20)
# activation, weights, biases
w = weights.pop()
b = biases.pop()
p = activations.pop()
a = activations.pop()
print(p.shape)

# convolution backprop
R_conv = backprop_conv(p, w, b, tf.reshape(R3, [1, 7, 7, 20]))
print(R_conv.shape)
R2 = backprop_pooling(a, R_conv)
print(R2.shape)

# 예제 7.11 𝑅2에서 역합성곱과 언풀링 과정을 수행하고 𝑅1 벡터를 구하는 코드

# layers = [hidden_1, pool_1]
# (1, 28, 28, 5)(1, 14, 14, 5)
# activation, weights, biases
w = weights.pop()
b = biases.pop()
p = activations.pop()
a = activations.pop()

# convolution backprop
R_conv = backprop_conv(p, w, b, R2)
print(R_conv.shape)

R1 = backprop_pooling(a, R_conv)
print(R1.shape)
print(np.sum(sess.run(R1)))

# 예제 7.12 𝑅1 결과에서 원본 이미지까지 LRP를 수행하는 코드

img_activations = getActivations(x_image, imageToUse)
w = weights.pop()
b = biases.pop()
R0 = backprop_conv(img_activations, w, b, R1)
print(f"R0:\n{R0}")
LRP_out = sess.run(R0)
print(f"LRP out:\n{LRP_out}")


# 예제 7.13 원본 이미지 형태로 타당성 전파를 수행하고 결과물을 이미지 형태로 출력하는 코드

plt.imshow(LRP_out.reshape(28, 28), interpolation="nearest", cmap=plt.cm.jet)

# 예제 7.13 합성곱 신경망 전체에 대해 LRP를 수행하는 코드

def getLRP(img):
    predict = sess.run(out_y, feed_dict={x: img.reshape([-1, 784]), keep_prob:1.0})[0]
    layers = [hidden_1, pool_1, hidden_2, pool_2, hidden_3]
    weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*weights.*')
    biases = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='.*biases.*')

    # layers = [hidden_1, pool_1, hidden_2, pool_2, hidden_3]
    activations = []
    for layer in layers:
        activations.append(getActivations(layer, img))

    # get f_x
    f_x = max(predict)

    # get R4
    predict[predict < 0] = 0
    R4 = predict

    # get R3
    a = activations.pop()
    w = weights.pop()
    b = biases.pop()
    R3 = backprop_dense(a.reshape(1,980), w, b, R4)

    # get R2
    w = weights.pop()
    b = biases.pop()
    p = activations.pop()
    a = activations.pop()
    R_conv = backprop_conv(p, w, b, tf.reshape(R3, [1, 7, 7, 20]))
    R2 = backprop_pooling(a, R_conv)

    # get R1
    w = weights.pop()
    b = biases.pop()
    p = activations.pop()
    a = activations.pop()
    R_conv = backprop_conv(p, w, b, R2)
    R1 = backprop_pooling(a, R_conv)

    # get R0
    img_activations = getActivations(x_image, img)
    w = weights.pop()
    b = biases.pop()
    R0 = backprop_conv(img_activations, w, b, R1)
    LRP_out = sess.run(R0)
    return LRP_out

# 예제 7.13 합성곱 신경망 전체에 대해 LRP를 수행하는 코드

# get MNIST dataset index dict
mnist_dict = {}
idx = 0
for i in mnist.test.labels:
    label = np.where(i == np.amax(i))[0][0]
    if mnist_dict.get(label):
        mnist_dict[label].append(idx)
    else:
        mnist_dict[label] = [idx]
    idx += 1

# get LRP
nums = []
for i in range(10):
    img_idx = mnist_dict[i][0]
    img = mnist.test.images[img_idx]
    lrp = getLRP(img)
    nums.append(lrp)

# plot images
plt.figure(figsize=(20,10))
for i in range(2):
    for j in range(5):
        idx = 5 * i + j
        plt.subplot(2, 5, idx + 1)
        plt.title('digit: {}'.format(idx))
        plt.imshow(nums[idx].reshape([28, 28]), cmap=plt.cm.jet)
        plt.colorbar(orientation='horizontal')
plt.tight_layout()
sess.close()
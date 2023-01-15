import numpy as np
import matplotlib.pyplot as plt
from pip._internal.network import session
from skimage.color import gray2rgb, rgb2gray
from skimage.util import montage as montage2d
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces() # 올리베티 얼굴 데이터베이스를 저장
                                # faces.data는 (400,4096) 크기의 튜플, 400은 데이터 개수, 4096은 64 * 64 이미지 벡터를 일렬로 늘어놓은 것

# make each image color so lime_image works correctly
X_vec = np.stack([gray2rgb(iimg) for iimg in faces.data.reshape((-1, 64, 64))],0) # faces.data의 이미지 한 장(iimg)에 대해 이미지를 (64,64) 크기로 재조정
# np.stack = np.array를 합치는 함수
# 재조정할데 reshape함수는 -1 파라미터를 사용하는데 -> (64 * 64) 크기에 맞춰 원본 이미지를 조정
# 이렇게 재조정된 이미지를 skimage의 gray2rgb 메서드에 입력. gray2rgb는 흑백 이미지를 RGB 3채널로 확장해줌
# 이렇게 이미지 전처리가 끝난 얼굴 데이터셋의 모양은 (400,64,64,3)으로, 각 원소는 순서대로 이미지 개수, 이미지 크기, 컬러 채널 수를 의미
y_vec = faces.target.astype(np.uint8)
# 이미지에 대응되는 사람 레이블을 정수형으로 저장

fig, ax1 = plt.subplots(1,1, figsize = (8,8))
ax1.imshow(montage2d(X_vec[:,:,:,0]), cmap='gray', interpolation = 'none') # montage2d는 직렬로 배열된 이미지를 격자 형태로 캔버스에 그리는 함수 , cmap은 컬러맵, interpolation은 이미지 해석이 쉽게 필터를 씌우는 코드
ax1.set_title('All Faces')
ax1.axis('off')

# 이미지 데이터 한 장을 그리는 코드
index = 93
plt.imshow(X_vec[index], cmap='gray')
plt.title('{} index face'.format(index))
plt.axis('off')

# sklearn 패키지에 있는 train_test_split 함수를 사용해서 X_vec과 y_vec으로부터 학습용과 테스트용 데이터세트를 분리하는 코드
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_vec, y_vec, train_size=0.70)

# MLP가 학습할 수 있도록 이미지 전처리를 수행하는 파이프라인 생성
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
class PipeStep(object):
    """ Wrapper for turning functions into pipeline transforms (no-fitting) """
    def __init__(self, step_func):
        self._step_func=step_func
    def fit(self,*args):
        return self
    def transform(self,X):
        return self._step_func(X)

makegray_step = PipeStep(lambda img_list: [rgb2gray(img) for img in img_list]) # rgb 채널이 3인 데이터를 흑백으로 바꿈
flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list]) # 64 * 64 크기의 이미지 데이터를 한 줄로 펼침. MLP는 1차원 배열만 처리할 수 있기 때문
# ravel은 다차원 배열을 1차원 배열로 해줌

# 흑백 처리와 1차원 배열 변환 과정을 하나로 wrapping
simple_pipeline = Pipeline([
    ('Make Gray', makegray_step),
    ('Flatten Image', flatten_step),
    ('MLP', MLPClassifier(
        activation='relu', #MLP는 과적합이 발생하기 쉬우므로 ReLU를 사용
        hidden_layer_sizes=(400, 40),
    random_state=1)) # 시드 값
])

# 학습 데이터를 MLP가 있는 파이프라인에 붓는 코드
simple_pipeline.fit(X_train, y_train)

# classification_report를 사용해서 모델 성능을 테스트하는 코드
pipe_pred_test = simple_pipeline.predict(X_test)
pipe_pred_prop = simple_pipeline.predict_proba(X_test)

from sklearn.metrics import classification_report # 테스트 데이터셋의 예측 결과(pipe_pred_test)와 실제 테스트 레이블(y_test)를 비교해서 정확도, 재현율, F1-점수를 낸 다음, 결과값을 평균
print(classification_report(y_true=y_test, y_pred = pipe_pred_test))

# Normalizer 전처리 과정을 추가해서 MLP를 학습시키는 코드
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neural_network import MLPClassifier

class PipeStep(object):
    """ Wrapper for turning functions into pipeline transforms (no-fitting) """
    def __init__(self, step_func):
        self._step_func=step_func
    def fit(self,*args):
        return self
    def transform(self,X):
        return self._step_func(X)


makegray_step = PipeStep(lambda img_list: [rgb2gray(img) for img in img_list])
flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])

simple_pipeline = Pipeline([
    ('Make Gray', makegray_step),
    ('Flatten Image', flatten_step),
    ('Normalize', Normalizer()),
    # add Normalizer preprocessing step
    ('MLP', MLPClassifier(
        activation='relu',
        hidden_layer_sizes=(400, 40),
        random_state=1)),
])

simple_pipeline.fit(X_train, y_train)


# 필자가 찾은 최적의 파이프라인 조합

simple_pipeline = Pipeline([ # Pipeline은 파라미터로 튜플들의 리스트를 받음
    ('Make Gray', makegray_step),
    ('Flatten Image', flatten_step),
    ('Normalize', Normalizer()),
    # add Normalizer preprocessing step
    ('MLP', MLPClassifier(
        activation='relu',
        alpha=1e-7, #L2규제를 적용하기 위한 매개변수
        epsilon=1e-6,
        hidden_layer_sizes=(800, 120),
        random_state=1)),
])

simple_pipeline.fit(X_train, y_train)

## 여기서부터 LIME 사용!!!!
# LIME의 이미지 설명체와 이미지 분할 알고리즘을 선언하는 코드
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm

explainer = lime_image.LimeImageExplainer()

# 이미지 분할 알고리즘: quickshift(기본), slic, felzenszwalb
segmenter = SegmentationAlgorithm(
        'slic',
        n_segments=100, # 이미지 분할 조각 개수
        compactness=1, # 분할한 이미지 조각으로부터 유사한 파트를 합치는 함수. 숫자가 클수록 합쳐지는 비율이 높음
        sigma=1) # 분할한 이미지를 부드럽게 깎아주는 정도. 0 ~ 1 (1에 가까울수록 부드럽게 깎아줌)

# 테스트 0번 이미지에 대해 설명 모델을 구축하는 코드
olivetti_test_index = 0

exp = explainer.explain_instance(
    X_test[olivetti_test_index],
    classifier_fn = simple_pipeline.predict_proba,
    top_labels=6, # simple_pipeline 모델이 예측한 1등부터 6등까지의 분류 값을 분석
    num_samples=1000, # 설명 모델이 결정 경계를 결정하기 위해 샘플링하는 공간의 크기
    segmentation_fn=segmenter)

# 올리베티 데이터 0번을 설명체에 통과시켜 XAI를 수행하는 코드

from skimage.color import label2rgb

# set canvas
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize = (8, 8))

# 예측에 도움이 되는 세그먼트 출력
temp, mask = exp.get_image_and_mask(
    y_test[olivetti_test_index],
    positive_only=True, #설명 모델이 가장 높게 분류한 라벨을 인식하는 데 도움이 되는 이미지 영역만 출력하는 파라미터. False를 하면, 모든 이미지 조각에 대한 마스킹 레이블을 출력
    num_features=8, # XAI에 사용하기 위한 분할 영역의 크기
    hide_rest=False) # 이미지를 분류하는 데 도움이 되는 서브모듈만 출력할지를 결정한다. 이 값이 True라면 XAI에 도움이 되는 영역을 제외하고 나머지 영역을 흑백으로 출력
# get_image_and_mask의 실행 결과는 1차원 튜플로, 이미지 분할이 완료된 원본 이미와 분할된 영역(masking area)을 반환
ax1.imshow(
        label2rgb(mask, temp, bg_label = 0), # 이미지 위에 형광색 마스킹을 해주는 함수. 원본 이미지 위에 마스킹 결과를 겹쳐서 출력
        interpolation = 'nearest')

ax1.set_title('Positive Regions for {}'.format(y_test[olivetti_test_index]))

# show all segments
temp, mask = exp.get_image_and_mask(
    y_test[olivetti_test_index],
    positive_only=False, # 모든 이미지 조각 마스킹 레이블을 출력
    num_features=8,
    hide_rest=False)

ax2.imshow(
    label2rgb(4 - mask, temp, bg_label = 0),
    interpolation = 'nearest')

ax2.set_title('Positive/Negative Regions for {}'.format(y_test[olivetti_test_index]))

# 설명 모델이 유용하게 사용한 이미지 조각만을 출력
ax3.imshow(temp, interpolation = 'nearest') # positive_only=False
ax3.set_title('Show output image only')

# get_image_and_mask의 분할 영역을 정수형 타입으로 시각화
ax4.imshow(mask, interpolation = 'nearest')
ax4.set_title('Show mask only')

# 올리베티 얼굴 테스트 데이터 0번(3번 인물)으로부터 추가 설명을 출력하는 코드

olivetti_test_index = 1

# now show them for each class
fig, m_axs = plt.subplots(2,6, figsize = (12,4))
for i, (c_ax, gt_ax) in zip(exp.top_labels, m_axs.T): # exp.top_labels: 설명 모델이 인물을 분류한 순서를 저정
    temp, mask = exp.get_image_and_mask(
            i,
            positive_only=True,
            num_features=12,
            hide_rest=False,
            min_weight=0.001)

    c_ax.imshow(
            label2rgb(mask,temp, bg_label = 0),
            interpolation = 'nearest')
    c_ax.set_title('Positive for {}\nScore:{:2.2f}%'.format(i, 100*pipe_pred_prop[olivetti_test_index, i])) # pipe_pred_prop은 설명 모델이 각 클래스를 예측한 확률을 저장

    c_ax.axis('off')

    face_id = np.random.choice(np.where(y_train==i)[0])

    gt_ax.imshow(X_train[face_id])
    gt_ax.set_title('Example of {}'.format(i))
    gt_ax.axis('off')
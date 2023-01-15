from sklearn.datasets import fetch_20newsgroups
import sklearn
import sklearn.metrics
from sklearn.naive_bayes import MultinomialNB

# 데이터 불러오기
newsgroups_train = fetch_20newsgroups(subset='train') # 학습용 데이터 저장
newsgroups_test = fetch_20newsgroups(subset='test') # 테스트용 데이터 저장

# 클래스 이름 줄이기
class_names = [x.split('.')[-1] if 'misc' not in x else '.'.join(x.split('.')[-2:])
               for x in newsgroups_train.target_names]
# 뉴스 그룹에 대한 20가지 카테고리

class_names[3] = 'pc.hardware'
class_names[4] = 'mac.hardware'
# class_names에 3,4번째 인덱스가 같아서 혼돈을 방지하기 위해 설정

# TF-IDF를 사용해서 문서를 숫자 벡터로 변환하는 전처리 과정
# 기계는 모든 문자를 이해할 수 없다. 사람이 사용하는 문자열을 기계도 알아들을 수 있는 형태로 변환하는 과정 (= 문자열 벡터화)
# TF - IDF는 여러 문서로 이루어진 문서군이 있을 때 어떤 단어가 특정 문서 내에서 얼마나 중요한 것인지를 나타내는 통계적 수치!
vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False) # True라면 모든 문자를 소문자로 변환
train_vectors = vectorizer.fit_transform(newsgroups_train.data) #학습 데이터를 벡터화하기 위해 fit_transform 메서드 호출
# fit_transform = fit + transform , fit은 뉴스 문자열을 벡터화하면서 벡터 공간을 효율적으로 설계하는 메서드 , transform은 fit으로 설계된 공간에 데이터를 옮기는 메서드
test_vectors = vectorizer.transform(newsgroups_test.data) # fit_transform을 하지 않는 이유는 train_vectors를 변형하면서 fit으로 벡터 공간을 완성했기 때문
# 여기서도 fit_transform하게 되면 새로운 문자열 벡터가 탄생하게 됨

#학습하기
nb = MultinomialNB(alpha=.01) # MultinomialNB는 다항분포 나이브 베이즈(Multinomial Naive Bayes)의 약자
# 다항분포 나이브 베이즈는 벡터 입력값에 대해 해당 문서가 특정 카테고리에 속할 확률을 계산
# 나이브 베이즈 모델은 입력된 기사가 기존에 학습된 모델의 단어 사용빈도와 비교했을 때 얼마나 가까운지 확률적으로 결과를 비교
# alpha를 파라미터로 받는데, 값이 너무 작으면 과적합, 너무 크면 학습이 잘 안됨
nb.fit(train_vectors , newsgroups_train.target) # 앞에 인자는 2차원 리스트로 만든 feature 데이터 , 뒤에 인자는 훈련데이터로 제공한 각 데이터셋을 사람이 판단하여 알려준 데이터 셋
# .target: Label 데이터, Numpy 배열로 이루어져 있습니다.
# fit은 LinearRegression 하기 위한 함수라고 생각하면 됨

#테스트하기
pred = nb.predict(test_vectors) # 테스트 데이터에서 다항분포 나이브 베이즈 모델이 뉴스 카테고리를 어떻게 예측하는지 측정한다
# 예측된 클래스의 목록을 반환. 모든 데이터 포인트에 대해 하나의 예측
# 예측된 값들을 pred에 저장
# predict 함수는 새로운 속성들을 넣었을 때 그 클래스에 속하는지 속하지 않는지를 나타내는 1 또는 0으로 구성된 벡터를 반환
sklearn.metrics.f1_score(newsgroups_test.target , pred, average='weighted') # 모델은 테스트 데이터를 예측하고 실제 결과와 비교해 F1-점수를 출력

print(sklearn.metrics.f1_score(newsgroups_test.target , pred, average='weighted')) # 83.5%의 성능으로 뉴스 카테고리를 분류

## 윗 과정은 모델을 생성하는 과정
## LIME을 적용하고 하이라이트 표시 기능 -> 파이피라인 기능을 통해 수행
## 파이프라인이란 우리가 만든 기능을 배관(pipe)처럼 이어서 적절한 순서대로 처리되게 하는 기능 -> 예를 들어, 모듈 A,B,C가 있을때 파이프라인은 호풀만으로도 모든 기능을 수행하도록 연결
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(vectorizer , nb) # vectorizer와 nb를 하나의 파이프라인으로 묶음

predict_classes = pipe.predict_proba([newsgroups_test.data[0]]).round(3)[0] # 테스트 데이터 인덱스 0번이 스무 가지 카테고리 중 어디에 속하는지에 관한 확률이 저장

print(predict_classes)

# 인덱스 0번이 어느 카테고리에 적합한지 순위로 보여줌
rank = sorted(range(len(predict_classes)), key = lambda i: predict_classes[i], reverse = True)
for i in rank:
    print('순위: [{:>5}]\t뉴스 카테고리 순서: {:<3}\t카테고리에 기사가 속할 가능성: class ({:.1%})'.format(rank.index(i)+1,i,predict_classes[i]))
    # list.index(a): 리스트 안에 a라는 인덱스가 몇번째에 있는지 반ㄴ환

## 제보 기사 카테고리 분류기 프로토타입을 완성
## 이제 LIME을 사용해서 뉴스 분류기 모델을 설명

# LIME 구현체는 기본적으로 텍스트 설명체, 이미지 설명체, 테이블 분류, 선형 공간 분류를 수행하게 미리 정의된 모듈과 사용자가 직접 수정할 수 있는 이산 모듈과 설명 모듈을 제공
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names = class_names) # LIME 텍스트 하이라이트 알고리즘 구현체
# 이 구현체는 파라미터로 피처를 선택하는 방식이나 BOW(Bag of Words) 알고리즘 수행 방식, 커널 크기 등을 수동으로 지정할 수 있다.
# 현재는 필수 파라미터(카테고리)만 넣음

exp =  explainer.explain_instance(newsgroups_test.data[0],pipe.predict_proba,top_labels=1)
# 0번 데이터 중 일부 벡터를 변형해서 분류기 출력 결과가 달라지는지 추적하고, 입력된 모델을 모사하는 선형 모델(linear model)을 만든다
# 이 선형 모델의 카테고리 분류 기준이 결정 경계가 되고, 결정 경계에 걸리는 0번 데이터의 단어 결합이 서브 모듈로 출력
#  explainer.explain_instance() 메서드는 최소 2가지의 파라미터가 필요. 해석하고 싶어 하는 데이터, 모델
# top_labels는 분류 가능성이 높은 클래스를 순서대로 몇 개를 보여줄지 결정하는 파라미터

print(exp.available_labels()) # LIME이 잘 작동하는지 확인하기 위한 메서드 , 입력된 데이터에 대해 설명이 가능한 레이블을 출력
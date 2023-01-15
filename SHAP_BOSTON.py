import shap
from sklearn.model_selection import train_test_split

X, y = shap.datasets.boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

X_train[:10]

# 예제 5.23 방의 개수와 집값 간의 관계를 산점도로 그리는 코드

# drawing scatter plot
import matplotlib.pylab as plt
import matplotlib

matplotlib.style.use('ggplot')

fig, ax1 = plt.subplots(1,1, figsize = (12,6))

ax1.scatter(X['RM'], y, color='black', alpha=0.6)
ax1.set_title('Relation # of Rooms with MEDV')
ax1.set_xlim(2.5, 9)
ax1.set_xlabel('RM')
ax1.set_ylim(0, 55)
ax1.set_ylabel('MEDV \n Price $1,000')

# 선형 모델을 이용해서 방 개수와 주택 가격 간의 관계를 구하는 코드

from sklearn import linear_model
import pandas as pd

linear_regression = linear_model.LinearRegression()
linear_regression.fit(X=pd.DataFrame(X_train['RM']), y=y_train)
# X_train : 방의 개수 , y_train : 주택 매매 가격
prediction = linear_regression.predict(X=pd.DataFrame(X_test['RM']))

print('a value: ', linear_regression.intercept_) # 절편을 반환. y=ax+b 식이 있을 때 b를 말함
print('b value: ', linear_regression.coef_) # array를 반환. 인덱스가 0번째 밖에 없음. 기울기를 알려줌
print('MEDV = {:.2f} * RM {:.2f}'.format(linear_regression.coef_[0], linear_regression.intercept_))

# 예제 5.39 방의 개수가 달라질 때 주택 매매 가격을 예측하는 그래프와 데이터를 한꺼번에 플롯으로 그리는 코드

# scatter Train, Test data with Linear Regression Prediction
# 선형 회귀 분석을 사용함. 회귀 분석은 데이터를 전부 반영하지는 않았지만, 데이터의 추세(trend)는 설명할 수 있음
# 방의 수에 따라 집값이 비례해서 증가함을 볼 수 있다
fig, ax1 = plt.subplots(1,1, figsize = (12,6))
ax1.scatter(X_train['RM'], y_train, color='black', alpha=0.4, label='data') # 학습용 데이터는 검정색으로 표현
ax1.scatter(X_test['RM'], y_test, color='#993299', alpha=0.6, label='data')
ax1.set_title('Relation # of Rooms with MEDV')
ax1.set_xlim(2.5, 9)
ax1.set_xlabel('RM')
ax1.set_ylim(0, 55)
ax1.set_ylabel('MEDV \n Price $1,000')
ax1.plot(X_test['RM'], prediction, color='purple', alpha=1, linestyle='--', label='linear regression line')
ax1.legend()

# 모델 예측치와 실제 집값 간의 RMSE를 구하는 코드
# 선형 회귀 분석으로 예측한 값은 오차가 많이 남. 성능을 측정하는 새로운 지표가 필요. 오차 지표는 실제 값과 예상 값 사이의 벡터 거리로 측정.
# -> 평균 제곱근 편차(Root Mean Square Deviation, 이하 RMSE) 사용
from sklearn.metrics import mean_squared_error
import numpy as np
rmse = np.sqrt(mean_squared_error(y_test, prediction))
print("RMSE: %f" % (rmse))

## 여기서부터 SHAP를 사용
# xgboost의 선형 회귀 모델로 주택 매매 가격을 예측하는 모델을 만들고 학습하는 코드
import xgboost

# train XGBoost model
model = xgboost.XGBRegressor(objective ='reg:linear') # 목적 함수를 선형 회귀로 설정
# XGBRegressor 패키지 중 회귀 분석과 관련된 모델을 모아놓은 클래스
model.fit(X_train, y_train)
preds = model.predict(X_test)

# 예제 5.42 전체 피처를 사용해서 학습시킨 모델의 RMSE를 구하는 코드
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
# 전체 피처를 사용해서 학습시킨 모델이, 피처 하나만 사용해서 학습한 모델보다 예측을 더 잘한다는 결론

# 예제 5.43 SHAP의 설명체를 정의하고 섀플리 값을 계산하는 로직

# SHAP 값으로 모델의 예측을 설명
shap.initjs() # 주피터 노트북에 자바스크립트 프리셋을 로드
# SHAP는 결괏값으로 보여주기 위해 자바스크립트 비주얼라이제이션 라이브러리를 사용

# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(model) # 새플리 값의 출력 형태를 앙상블 트리 형태로 시각화
# 설명체의 종류는 모델에 따라 선언. 일반적으로 딥러닝 모델 -> DeepExplainer
# 그 밖의 모델에 대한 트리 설명체(TreeExplainer) -> 각 피처가 개별적이고, 피처 간 조합이 결과에 영향을 미칠 경우 사용
# 각 피처에 대한 가중치를 설정할 수 있는 커널 설명체(KernelExplainer)
shap_values = explainer.shap_values(X_train) # 새플리 값을 계산
# 13개의 피처에 대한 404개의 값을 반환

# visualize the first prediction's explanation
# (use matplotlib=True to avoid Javascript)
shap.force_plot(explainer.expected_value, shap_values[0,:], X_train.iloc[0,:]) # 특정 데이터에 대한 새플리 값을 상세하게 분해하고 시각화

# 예제 5.44 259번 데이터에 대해서 방의 개수(RM)와 집 가격(MEDV)이 어떤 관계가 있는지 플롯으로 그리는 코드

fig, ax1 = plt.subplots(1,1, figsize = (12,6))

idx = 259
ax1.scatter(X['RM'], y, color='black', alpha=0.6)
ax1.scatter(X_train['RM'].iloc[idx], y_train[idx], c='red', s=150)
ax1.set_title('Relation # of Rooms with MEDV')
ax1.set_xlim(2.5, 9)
ax1.set_xlabel('RM')
ax1.set_ylim(0, 55)
ax1.set_ylabel('MEDV \n Price $1,000')

# 예제 5.45 데이터 259번에 대한 섀플리 영향도를 그리는 코드

# load JS visualization code to notebook
shap.initjs()
# 259번째 인덱스의 섀플리 영향도를 시각화
shap.force_plot(explainer.expected_value, shap_values[259,:], X_train.iloc[259,:])

# load JS visualization code to notebook
shap.initjs()

# 모델이 학습 데이터를 예측한 결과에 대해 SHAP 분석한 결과를 출력
shap.force_plot(explainer.expected_value, shap_values, X_train)

# 방 개수 피처가 집값에 미치는 섀플리 영향도를 시각화하는 플롯
shap.dependence_plot("RM", shap_values, X_train)

# 예제 5.48 전체 피처들이 섀플리 값 결정에 어떻게 관여하는지 시각화하는 코드

# 모든 피처에 대해 SHAP 값을 계산하고, 영향력을 시각화하는 코드
shap.summary_plot(shap_values, X_train)

# 예제 5.49 피처별 섀플리 값을 막대 타입으로 비교하는 코드
shap.summary_plot(shap_values, X_train, plot_type="bar")

# 예제 5.50 xgboost의 피처 중요도를 호출하는 코드
xgboost.plot_importance(model)
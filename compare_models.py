# 앙상블 : 소프트 보팅 (의사 결정 트리 , 랜덤 포레스트)

from joblib import dump,load
from random_forest import best_random_forest_model
from decision import best_model
import pandas as pd
from sklearn.model_selection import train_test_split


# 전처리된 데이터 가져오기
data = pd.read_excel('processed_data.xlsx')

# 특성과 타겟 변수 선택
X = data[['gre', 'gpa']]
y = data['admit']

# 데이터 분할 (학습 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 의사 결정 트리 모델 저장
dump(best_model, 'decision.joblib')

# 랜덤 포레스트 모델 저장
dump(best_random_forest_model, 'random_forest.joblib')

# 의사 결정 트리 모델 불러오기
decision_tree_model = load('decision.joblib')

# 랜덤 포레스트 모델 불러오기
random_forest_model = load('random_forest.joblib')

# 의사 결정 트리 모델의 예측 확률
y_prob_decision_tree = best_model.predict_proba(X_test)

# 랜덤 포레스트 모델의 예측 확률
y_prob_random_forest = best_random_forest_model.predict_proba(X_test)


import numpy as np

# 의사 결정 트리와 랜덤 포레스트 모델의 예측 확률 평균 계산
y_prob_combined = (y_prob_decision_tree + y_prob_random_forest) / 2

# 평균 확률 중에서 가장 높은 클래스 선택
y_pred_combined = np.argmax(y_prob_combined, axis=1)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 각 개별 모델의 예측
y_pred_decision_tree = decision_tree_model.predict(X_test)
y_pred_random_forest = random_forest_model.predict(X_test)

# 각 개별 모델의 성능 평가
accuracy_decision_tree = accuracy_score(y_test, y_pred_decision_tree)
accuracy_random_forest = accuracy_score(y_test, y_pred_random_forest)

# 앙상블 모델의 예측 (소프트 투표)
y_pred_ensemble = (y_pred_decision_tree + y_pred_random_forest) // 2

# 앙상블 모델의 성능 평가
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)

# 각 모델의 혼동 행렬 출력
conf_matrix_decision_tree = confusion_matrix(y_test, y_pred_decision_tree)
conf_matrix_random_forest = confusion_matrix(y_test, y_pred_random_forest)
conf_matrix_ensemble = confusion_matrix(y_test, y_pred_ensemble)

# 결과 출력
print("Decision Tree Model Accuracy:", accuracy_decision_tree)
print("Random Forest Model Accuracy:", accuracy_random_forest)
print("Ensemble Model Accuracy:", accuracy_ensemble)

# 혼동 행렬 시각화
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 6))

plt.subplot(131)
sns.heatmap(conf_matrix_decision_tree, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(132)
sns.heatmap(conf_matrix_random_forest, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.subplot(133)
sns.heatmap(conf_matrix_ensemble, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Ensemble Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

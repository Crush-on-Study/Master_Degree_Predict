import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# 전처리된 데이터 가져오기
data = pd.read_excel('processed_data.xlsx')

# 특성과 타겟 변수 선택 (예시: 'gre'와 'gpa'를 특성으로 사용)
X = data[['gre', 'gpa']]
y = data['admit']

# 데이터 분할 (학습 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 하이퍼파라미터 그리드 설정
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# 랜덤 포레스트 모델 초기화
random_forest_model = RandomForestClassifier(max_features='sqrt')  # 'sqrt'를 사용하거나 다른 유효한 값 선택


# 그리드 서치 객체 초기화
grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=5, n_jobs=-1)

# 그리드 서치로 모델 학습 및 튜닝
grid_search.fit(X_train, y_train)

# 최적 하이퍼파라미터 출력
print("Best Hyperparameters:")
print(grid_search.best_params_)

# 최적 모델 선택
best_random_forest_model = grid_search.best_estimator_

# 테스트 데이터로 예측
y_pred = best_random_forest_model.predict(X_test)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# ROC 곡선 및 AUC 계산
y_prob = best_random_forest_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# 결과 데이터 출력
print("\nRandom Forest Model (Tuned Hyperparameters):")
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')

# 시각화: ROC 곡선 그래프
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (Random Forest)')
plt.legend(loc='lower right')
plt.show()

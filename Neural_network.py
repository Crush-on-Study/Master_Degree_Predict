import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# 전처리된 데이터 가져오기
data = pd.read_excel('processed_data.xlsx')

# 특성과 타겟 변수 선택
X = data[['gre', 'gpa']]
y = data['admit']

# 데이터 분할 (학습 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 클래스 불균형 처리: 언더샘플링
undersampler = RandomUnderSampler(sampling_strategy='majority')
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# 신경망 (MLP) 모델 초기화 및 학습
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, alpha=0.001, random_state=42)  # 규제 적용
mlp_model.fit(X_train_resampled, y_train_resampled)

# 테스트 데이터로 예측
y_pred = mlp_model.predict(X_test)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# ROC 곡선 및 AUC 계산
y_prob = mlp_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# 결과 데이터
print("Neural Network Model:")
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
plt.title('Receiver Operating Characteristic (ROC) Curve (Neural Network)')
plt.legend(loc='lower right')
plt.show()

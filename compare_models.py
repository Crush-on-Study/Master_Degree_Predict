import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# 전처리된 데이터 가져오기 (데이터 파일 이름에 따라 변경)
data = pd.read_excel('processed_data.xlsx')

# 특성 선택 (gre, gpa, rank)
input_data = data[['gre', 'gpa', 'rank']]

# 각 모델 초기화
logistic_model = LogisticRegression()
decision_tree_model = DecisionTreeClassifier()
random_forest_model = RandomForestClassifier()
svm_model = SVC(probability=True)
mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)

# 각 모델에 입력 데이터 제공하여 학습
logistic_model.fit(input_data, data['admit'])
decision_tree_model.fit(input_data, data['admit'])
random_forest_model.fit(input_data, data['admit'])
svm_model.fit(input_data, data['admit'])
mlp_model.fit(input_data, data['admit'])

# 각 모델의 예측 결과 계산
logistic_admit = logistic_model.predict(input_data)
decision_tree_admit = decision_tree_model.predict(input_data)
random_forest_admit = random_forest_model.predict(input_data)
svm_admit = svm_model.predict(input_data)
mlp_admit = mlp_model.predict(input_data)

# 각 모델의 신뢰도 확인
logistic_proba = logistic_model.predict_proba(input_data)
decision_tree_proba = decision_tree_model.predict_proba(input_data)
random_forest_proba = random_forest_model.predict_proba(input_data)
svm_proba = svm_model.predict_proba(input_data)
mlp_proba = mlp_model.predict_proba(input_data)

# 각 모델의 예측 결과 출력
print("로지스틱 회귀 모델 예측 결과 (admit):", logistic_admit)
print("결정 트리 모델 예측 결과 (admit):", decision_tree_admit)
print("랜덤 포레스트 모델 예측 결과 (admit):", random_forest_admit)
print("서포트 벡터 머신 (SVM) 모델 예측 결과 (admit):", svm_admit)
print("신경망 (Neural Network) 모델 예측 결과 (admit):", mlp_admit)

# 각 모델의 신뢰도 출력
print("\n로지스틱 회귀 모델 신뢰도:\n", logistic_proba)
print("결정 트리 모델 신뢰도:\n", decision_tree_proba)
print("랜덤 포레스트 모델 신뢰도:\n", random_forest_proba)
print("서포트 벡터 머신 (SVM) 모델 신뢰도:\n", svm_proba)
print("신경망 (Neural Network) 모델 신뢰도:\n", mlp_proba)

# 이제 여기다가 본인 스펙 넣으세요
def input_your_data():
    # 추가로 입력 데이터를 설정하고 각 모델에 입력한 후 예측 결과를 확인하는 코드 추가
    input_data = [[0, 3.64, 1]]  # 원하는 값을 넣기

    # 각 모델의 예측 결과 계산
    logistic_admit = logistic_model.predict(input_data)
    decision_tree_admit = decision_tree_model.predict(input_data)
    random_forest_admit = random_forest_model.predict(input_data)
    svm_admit = svm_model.predict(input_data)
    mlp_admit = mlp_model.predict(input_data)

    # 각 모델의 신뢰도 확인
    logistic_proba = logistic_model.predict_proba(input_data)
    decision_tree_proba = decision_tree_model.predict_proba(input_data)
    random_forest_proba = random_forest_model.predict_proba(input_data)
    svm_proba = svm_model.predict_proba(input_data)
    mlp_proba = mlp_model.predict_proba(input_data)

    # 각 모델의 예측 결과 출력
    print("\n로지스틱 회귀 모델 예측 결과 (admit):", logistic_admit)
    print("결정 트리 모델 예측 결과 (admit):", decision_tree_admit)
    print("랜덤 포레스트 모델 예측 결과 (admit):", random_forest_admit)
    print("서포트 벡터 머신 (SVM) 모델 예측 결과 (admit):", svm_admit)
    print("신경망 (Neural Network) 모델 예측 결과 (admit):", mlp_admit)

    # 각 모델의 신뢰도 출력
    print("\n로지스틱 회귀 모델 신뢰도:\n", logistic_proba)
    print("결정 트리 모델 신뢰도:\n", decision_tree_proba)
    print("랜덤 포레스트 모델 신뢰도:\n", random_forest_proba)
    print("서포트 벡터 머신 (SVM) 모델 신뢰도:\n", svm_proba)
    print("신경망 (Neural Network) 모델 신뢰도:\n", mlp_proba)

import pandas as pd

data = pd.read_excel('gpascore.xlsx')

# 데이터 정보 확인
# print(data.info())

# 결측치가 있는 행 제거
data.dropna(subset=['gre'], inplace=True)

# 결측치 처리 후 데이터 정보 확인
# print(data.info())

data.to_excel('processed_data.xlsx',index=False)
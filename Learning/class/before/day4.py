# 1. 데이터프레임 기본 개념과 생성
import pandas as pd

# Series 생성
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
print(s)

# DataFrame 생성
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Score': [85, 90, 95]
}
df = pd.DataFrame(data)
print(df)



df_emp = pd.DataFrame(
    {
        '번호': [1001, 1002, 1003, 1004, 1005],
        '이름': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        '부서번호': [10, 20, 10, 30, 40],
        '입사일': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12'],   
        '전화': ['123-456-7890', '987-654-3210', '555-123-4567', '111-222-3333', '444-2548-5412']
    }
)
df_emp



import pandas as pd
import numpy as np

df = pd.DataFrame({
    '이름': ['홍길동', '김철수', '이영희'],
    '점수': [90, np.nan, 85]
})

print(df)

# 평균으로 결측치 대치
df['점수'] = df['점수'].fillna(df['점수'].mean())

print("\n 평균값으로 대치한 후:")
print(df)


# duplicated()
import pandas as pd

# 예제 DataFrame
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Alice', 'Charlie', 'Bob'],
    'Age': [25, 30, 25, 35, 30],
    'Score': [90, 85, 90, 88, 85]
})

print("원본 DataFrame:")
df


print("중복 여부 확인 (전체 행 기준):") # 행에 속한 컬럼값이 모두 다른 행과 일치할 때
print(df.duplicated())


# 중복 행 제거 (drop_duplicates())
df_no_dup = df.drop_duplicates()  # 첫번째 데이터는 유지
print("중복 제거 후:")
print(df_no_dup)


# 특정 열만 기준으로 중복 제거
df_no_dup_name = df.drop_duplicates(subset=['Name'])
print("Name 컬럼만 기준으로 중복 제거:")
print(df_no_dup_name)


# 병합과 조인
# 여러 DataFrame 합치기, merge와 concat의 차이 이해, concat(), merge(), join()

df1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})  # 이름을 가진 DF
df2 = pd.DataFrame({'ID': [1, 2], 'Score': [90, 80]})         # 점수를 가진 DF

# merge
merged = pd.merge(df1, df2, on='ID')  # 이름과 점수를 합칠 때 ID 컬럼을 기준으로 
merged


# merge
merged = pd.merge(df1, df2, on='ID')  # 이름과 점수를 합칠 때 ID 컬럼을 기준으로 
merged


# 중복된 index가 있는 경우
import pandas as pd

# 첫 번째 DataFrame
df1 = pd.DataFrame({
    'Name': ['Alice', 'Bob'],
    'Score': [90, 85]
}, index=['s1', 's2'])

# 두 번째 DataFrame
df2 = pd.DataFrame({
    'Name': ['Charlie', 'Bob'],
    'Score': [88, 82]
}, index=['s2', 's3'])

print("df1:")
print(df1)

print("\ndf2:")
print(df2)


# pd.concat()으로 단순 합치기 (인덱스 유지)
result = pd.concat([df1, df2])
print("\n단순 concat 결과 (중복 인덱스 허용):")
print(result)


# 인덱스를 무시하고 합치기 (ignore_index=True): 새로 0부터 인덱스 부여
result = pd.concat([df1, df2], ignore_index=True)
print("\n인덱스를 무시하고 합치기:")
print(result)


# 중복 인덱스 제거 (drop_duplicates() 사용)
result = pd.concat([df1, df2])
result = result[~result.index.duplicated(keep='first')]  # ~ : 부정, 중복되면 True, 첫번째는 False
print("\n중복된 인덱스를 제거 (첫 번째만 유지):")
print(result)


# 인덱스 재정의 후 합치기
# df2의 인덱스를 변경해서 충돌 방지
df2_reset = df2.reset_index(drop=True)

result = pd.concat([df1, df2_reset], ignore_index=True)
print("\n 인덱스를 재설정한 후 합치기:")
print(result)


# 인덱스 재정의 후 합치기
# df2의 인덱스를 변경해서 충돌 방지
df2_reset = df2.reset_index(drop=True)

result = pd.concat([df1, df2_reset], ignore_index=True)
print("\n 인덱스를 재설정한 후 합치기:")
print(result)


# join()을 이용한 합치기(인덱스를 기준으로 합치기)
import pandas as pd

# 왼쪽 DataFrame
df1 = pd.DataFrame({
    '이름': ['홍길동', '김철수', '이영희'],
    '수학': [90, 85, 88]
})
df1.set_index('이름', inplace=True)

# 오른쪽 DataFrame
df2 = pd.DataFrame({
    '이름': ['홍길동', '이영희', '박지민'],
    '영어': [95, 80, 75]
})
df2.set_index('이름', inplace=True)

# join 사용 (기본은 left join)
result = df1.join(df2)

print(result)



# inner join: 양쪽에 모두 존재하는 인덱스만 유지
df1.join(df2, how='inner')


# outer join: 모든 인덱스를 포함, 없는 값은 NaN
df1.join(df2, how='outer')



# 컬럼 이름이 중복되는 경우
df3 = pd.DataFrame({
    '수학': [70, 60, 50]
}, index=['홍길동', '김철수', '이영희'])

# 이름이 같은 컬럼이 충돌하므로 접미사 사용 필요
df1.join(df3, lsuffix='_왼쪽', rsuffix='_오른쪽')



# Matplotlib 시각화 예
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
url = "https://raw.githubusercontent.com/plotly/datasets/master/tesla-stock-price.csv"
df = pd.read_csv(url)

# 컬럼명 확인
print(df.columns)  # ['date', 'open', 'high', 'low', 'close', 'volume']

# 날짜를 datetime 형식으로 변환
df['date'] = pd.to_datetime(df['date'])

# 인덱스를 날짜로 설정
df.set_index('date', inplace=True)

# 종가(close) 시각화
df_early = df[df.index < '2020-01-01']

plt.figure(figsize=(12, 6))
plt.plot(df_early['close'], label='Close Price (2010~2019)', color='orange')
plt.title('Tesla Close Price (2010~2019)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
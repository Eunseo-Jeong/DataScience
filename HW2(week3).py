import pandas as pd
import numpy as np

array = [[3, '?', 2, 5], ['*', 4, 5, 6], ['+', 3, 2, '&'], [5, '?', 7, '!']]
df = pd.DataFrame(array)
df
#DataFrame 만들기


df.replace({"?":np.nan,"*":np.nan,"+":np.nan,"!":np.nan,"&":np.nan},inplace=True)
df
#DataFrame replace and print

df.isna().any() #df NaN 하나라도 있으면 True

df.isna().sum() #df NaN의 개수

df.dropna(axis=0,how='all',inplace=False)
#row 기준으로 모두 NaN이면 row 삭제, df에 적용 안함

df.dropna(axis=0,how='any',inplace=False)
#row 기준으로 하나라도 NaN이면 row 삭제, df에 적용 안함

df.dropna(axis=1,how='all',inplace=False)
#column 기준으로 모두 NaN이면 column 삭제, df에 적용 안함

df.dropna(axis=1,how='any',inplace=False)
#column 기준으로 하나라도 NaN이면 column 삭제, df에 적용 안함

df.dropna(axis=0,thresh=1,inplace=False)
#row 기준으로 NaN이 1개이상이면 row 삭제, df에 적용 안함



df.dropna(axis=0,thresh=2,inplace=False)
#row 기준으로 NaN이 2개 이상이면 row 삭제, df에 적용 안함

df.dropna(axis=1,thresh=1,inplace=False)
#column 기준으로 NaN이 1개 이상이면 column 삭제, df에 적용안함

df.dropna(axis=1,thresh=2,inplace=False)
#column 기준으로 NaN이 2개 이상이면 column 삭제, df에 적용안함

df.fillna(100) #NaN에 100채우기

mean=df[0].mean()
df[0].fillna(mean)
# NaN에 df[0]의 평균 넣기

mean=df[1].mean()
df[1].fillna(mean)
# MaN에 df[1]의 평균 넣기

mean=df[2].mean()
df[2].fillna(mean)
# NaN에 df[2]의 평균 넣기 (NaN없음)

mean=df[3].mean()
df[3].fillna(mean)
# NaN에 df[3]의 평균 넣기

df[0].fillna(df[0].median())
#NaN에 df[0].median 넣기

df[1].fillna(df[1].median())
#NaN에 df[1].median 넣기

df[2].fillna(df[2].median())
#NaN에 df[2].median 넣기 (NaN없음)

df[3].fillna(df[3].median())
#NaN에 df[3].median 넣기

df.ffill()
#NaN에 바로 앞에 있는 column 넣기

df.bfill()
#NaN에 바로 뒤에 있는 column 넣기

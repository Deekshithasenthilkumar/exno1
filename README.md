# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
       import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
WhatsApp Image 2025-03-12 at 15 45 36_7a2d00c4

df.shape
WhatsApp Image 2025-03-12 at 15 46 49_86fe44b4

df.describe()
WhatsApp Image 2025-03-12 at 15 47 58_2ecf0704

df.info()
WhatsApp Image 2025-03-12 at 15 49 09_2da1b363

df.head(5)
WhatsApp Image 2025-03-12 at 15 49 32_1417b103

df.tail(2)
WhatsApp Image 2025-03-12 at 15 50 01_8191d5ef

df.dropna(how='any').shape
WhatsApp Image 2025-03-12 at 15 50 53_c1421bd1

df.isnull().sum()
WhatsApp Image 2025-03-12 at 15 51 46_d0c3ad2c

mn=df.TOTAL.mean()
df.TOTAL.fillna(mn,inplace=True)
df
WhatsApp Image 2025-03-12 at 15 53 51_17625229

df.M1.dropna(inplace=True)
df
WhatsApp Image 2025-03-12 at 15 54 59_0f0a55df

df.isna().sum()
WhatsApp Image 2025-03-12 at 15 55 55_04c2eeeb

df['M1'].fillna(method='ffill',inplace=True)
WhatsApp Image 2025-03-12 at 15 56 40_cae15bc0

df.duplicated()
WhatsApp Image 2025-03-12 at 15 58 04_5449eff8

df['DOB']
WhatsApp Image 2025-03-12 at 15 59 07_82a53dca

import seaborn as sns
sns.heatmap(df.isnull(),yticklabels=False,annot=True)
WhatsApp Image 2025-03-12 at 15 59 51_fc0c894b

import pandas as pd
import seaborn as sns
import numpy as np
age=[1,3,28,27,25,92,30,39,40,50,26,24,29,94]
af=pd.DataFrame(age)
af
WhatsApp Image 2025-03-12 at 16 00 44_f2ba4106

sns.boxplot(data=af)
WhatsApp Image 2025-03-12 at 16 01 30_17a7c6f3

sns.scatterplot(data=af)
WhatsApp Image 2025-03-12 at 16 02 03_436ae0d0

q1=af.quantile(0.25)
q2=af.quantile(0.5)
q3=af.quantile(0.75)
iqr=q3-q1
iqr
WhatsApp Image 2025-03-12 at 16 02 36_3a8f9dc3

Q1=np.percentile(af,25)
Q3=np.percentile(af,75)
IQR=Q3-q1
IQR
WhatsApp Image 2025-03-12 at 16 04 01_f174e52f

lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
lower_bound
WhatsApp Image 2025-03-12 at 16 04 36_38ed6887

upper_bound
WhatsApp Image 2025-03-12 at 16 05 14_bb4f4d8a

outliers = [x for x in age if (x < lower_bound.iloc[0]) or (x > upper_bound.iloc[0])]
print("q1",q1)
print("q3",q3)
print("iqr",iqr)
print("lower bound",lower_bound)
print("upper bound",upper_bound)
print("outliers",outliers)
WhatsApp Image 2025-03-12 at 16 05 48_0c6a4c82

af=af[((af>=lower_bound)&(af<=upper_bound))]
WhatsApp Image 2025-03-12 at 16 09 26_8c4c02e0

data=[1,2,2,2,3,1,1,15,2,2,2,3,1,1,2]
mean=np.mean(data)
std=np.std(data)
print('mean of the dataset is',mean)
print('std.deviation is',std)
WhatsApp Image 2025-03-12 at 16 10 04_5ffe68a9

threshold=3
outlier=[]
for i in data:
  z=(i-mean)/std
  if z>threshold:
    outlier.append(i)
    print('Outlier in dataset is:',outlier)
WhatsApp Image 2025-03-12 at 16 10 20_7f599808

import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
data={'weight':[12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,
                66,69,202,72,75,78,81,84,232,87,90,93,96,99,258]}

df=pd.DataFrame(data)
df
WhatsApp Image 2025-03-12 at 16 11 37_474dd4e9

z=np.abs(stats.zscore(df))
print(df[z['weight']>3])
WhatsApp Image 2025-03-12 at 16 12 10_8bc9f6c8

val=[12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,60,63,
                66,69,202,72,75,78,81,84,232,87,90,93,96,99,258]


import numpy as np
out=[]
def d_o(val):
  ts=3
  m=np.mean(val)
  sd=np.std(val)
  for i in val:
    z=(i-m)/sd
    if np.abs(z)>ts:
      out.append(i)
  return out

op=d_o(val)
op
WhatsApp Image 2025-03-12 at 16 12 50_dfaf5648
# Result
Thus we have cleaned the data and removed the outliers by detection using IQR and Z-score method

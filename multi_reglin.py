import pandas as pd
import numpy as np
from sklearn import linear_model
import math

df=pd.read_csv("csv/homeprices.csv")

median_bedrooms=math.floor(df.bedrooms.median())
df.bedrooms=df.bedrooms.fillna(median_bedrooms)

reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)
input_data=pd.DataFrame([[300,3,30]], columns=['area', 'bedrooms', 'age'])    
print(reg.predict(input_data))


import pickle
with open('csv/reg_pickle','wb') as f:
    pickle.dump(reg,f)
    


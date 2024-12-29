import pandas as pd
import numpy as np
from sklearn import linear_model
df=pd.read_csv("homeprices.csv")

import math
median_bedrooms=math.floor(df.bedrooms.median())
df.bedrooms=df.bedrooms.fillna(median_bedrooms)

reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)
print(reg.predict([[300,3,30]]))
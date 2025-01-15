import pickle
with open('csv/gradient_coefficients.pkl', 'rb') as f:
    loaded_coefficients = pickle.load(f)

print(loaded_coefficients)


import pandas as pd

with open('csv/reg_pickle','rb') as f:
    mp=pickle.load(f)
input_data=pd.DataFrame([[300,3,30]], columns=['area', 'bedrooms', 'age'])    
print(mp.predict(input_data))

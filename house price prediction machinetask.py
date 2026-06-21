import pandas as pd 
from sklearn.linear_model import LinearRegression

data = {
    "Area" : [800,1200,2300,3500,2600],
    "Price" :[45000,75000,100000,125000,134000]
}

df = pd.DataFrame(data)

x = df[["Area"]]
y = df[["Price"]]

model = LinearRegression()
model.fit(x,y)

predict_price = model.predict([[1900]])

print("predicted price is :", predict_price[0])
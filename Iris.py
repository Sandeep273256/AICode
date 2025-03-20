import pandas as pd
from pandas import  read_csv

df = read_csv("iris.csv")
print(df.head())
print(df.tail())
print(df.columns)
from pandas import read_csv

df = read_csv('data.csv')

train = df.iloc[0:8040]
validation = df.iloc[8040:10050]
test = df.iloc[10050:12563]

labelName = 'area_type'
labelNumericName = 'price in rupees'

labels = ['availability','bedrooms','total_sqft','bath','balcony','ranked','price in rupees', 'area_type']

labels_cls = ['availability','bedrooms','total_sqft','bath','balcony','ranked','price in rupees']
labels_rgr = ['area_type','availability','bedrooms','total_sqft','bath','balcony','ranked']

labelIndex = df.columns.get_loc(labelName)

labelNumericIndex = df.columns.get_loc(labelNumericName)

cls_x = 'B' 
cls_y = 'P'
import pandas as pd
from sklearn.linear_model import LinearRegression

#Veri biliminde preprocessing (ön işleme), ham veriyi bir makine öğrenmesi modelinin anlayabileceği 
# ve işleyebileceği temiz, düzenli bir formata getirme sürecidir.  Audi projesinde yaptığım 
# "L" harfini silme, verileri sayıya çevirme ve gereksiz sütunları atma işlemleri bu sürecin birer parçasıdır.

df = pd.read_csv('Audi_A1_listings.csv')
print(df.head(3))

df = df.drop(columns=['index', 'href', 'MileageRank', 'PriceRank','PPYRank', 'Score']) #sildim

print(df.head(3))

df['Engine'] = df['Engine'].str.replace('L', '') #preprocessing
df['Engine'] = pd.to_numeric(df['Engine'])
print(df['Engine'])
print(df.head(3)) #hala kategorisel veriler var, bunları numeric veriye çevirmemiz gerek


df = pd.get_dummies(df, columns=['Type', 'Transmission', 'Fuel'], drop_first=True) #kategorisel veriler numeric veriye çevrildi, artık modelleme yapabiliriz
print(df.head(3)) 
#print(df.info())

y = df[['Price(£)']]
x = df.drop('Price(£)', axis=1) #axis = 1 sütünlar axis = 0 satırlar
lm = LinearRegression()
model = lm.fit(x.values, y)

print(model.predict([[2017, 3000, 1.6, 110, 1, 2600, 0, 1]]))

print(model.score(x.values, y))
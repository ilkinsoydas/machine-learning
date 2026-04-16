from prophet import Prophet
import yfinance as yf
import matplotlib.pyplot as plt

df = yf.download("BTC-USD", "2015-01-01", "2026-01-01")

df = df[['Close']]
df = df.reset_index()

df.columns = ['ds', 'y']
model = Prophet()

model.fit(df)
gelecek = model.make_future_dataframe(360)

tahmin = model.predict(gelecek)
print(tahmin[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5))
grafik = model.plot(tahmin)
model.plot_components(tahmin)
plt.show()

#ARMA, LSTM vs bakabilirsin daha detayli tahminler icin






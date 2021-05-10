import investpy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# filtering and ploting date, closing values

df = investpy.get_stock_historical_data(stock='BATA',
                                        country='india',
                                        from_date='01/01/2010',
                                        to_date='01/12/2020')[["Close"]] 



df.plot(figsize = (15,8))
plt.title('Historical Stock Value')
plt.xlabel('year')
plt.ylabel('closing value')
plt.show()
import numpy as np
import pandas as pd

# 1. İçinde eksik (NaN) değerler barındıran veri seti oluşturma (7 satır)
data = {
    'Name': ['Efe', 'Ali', 'Ayse', 'Ece', 'Mustafa', 'Can', 'Zeynep'],
    'Ages': [20, np.nan, 25, 16, 50, np.nan, 22],
    'Salary': [28000, 500, np.nan, 5000, 70000, 12000, np.nan],
    'City': ['Bursa', 'Ankara', 'Izmir', 'Istanbul', 'Istanbul', 'Bursa', 'Ankara']
}

df = pd.DataFrame(data)

# print("--- Orijinal DataFrame ---")
# print(df)
# print("\n")

# print(df.isna().sum())

# meanAge = round(df['Ages'].mean())
# df['Ages'] = df['Ages'].fillna(meanAge)
# print(df)
  
# df = df.dropna(subset = 'Salary')
# print(df)

  

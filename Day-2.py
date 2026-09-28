import pandas as pd

data = {
  'Name': ['Efe', 'Ali', 'Ayse', 'Ece', 'Mustafa' ],
  'Ages': [20, 10, 25, 16, 50 ],
  'Salary': [28000, 500, 34000, 5000, 70000],
  'City': ['Bursa', 'Ankara', 'Izmir', 'Istanbul', 'Istanbul']
}

df = pd.DataFrame(data)
# print(df['Salary'].mean())

# print(df.head(3))

# print(df.loc[df['Ages'] > 30])

# print(df.iloc[1:3])
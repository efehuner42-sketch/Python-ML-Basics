import pandas as pd

# 1. Adım: İçinde metinsel kategorik veriler barındıran basit bir DataFrame oluşturma
veri = {
    'Isim': ['Ahmet', 'Zeynep', 'Mehmet', 'Ayse', 'Can'],
    'Yas': [22, 20, 23, 21, 22],
    'Sehir': ['Istanbul', 'Bursa', 'Ankara', 'Istanbul', 'Bursa'],
    'Cinsiyet': ['Erkek', 'Kadin', 'Erkek', 'Kadin', 'Erkek']
}

df = pd.DataFrame(veri)

print("Orijinal DataFrame:")
print(df)
print("-" * 40)

yeni_df = pd.get_dummies(df, columns=["Sehir", "Cinsiyet"], dtype=int)

print(yeni_df)
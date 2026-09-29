import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1-Seaborn kütüphanesinin içinde hazır olarak gelen meşhur restoran bahşişleri (tips) veri setini yükle:

df =  sns.load_dataset('tips')

# 2-Veri setinin ilk 5 satırını ekrana yazdırarak (df.head()) sütunları incele (Hesap tutarı, bahşiş, cinsiyet, gün vb.).

print(df.head(5))

# 3-Toplam hesap (total_bill) ile bırakılan bahşiş (tip) arasındaki ilişkiyi gösteren bir grafik çizdir.

sns.scatterplot(x = "total_bill", y = "tip", data = df)
plt.title("Relationship")
plt.show()

# 4-Hangi günlerde daha fazla hesap ödendiğini görmek için X ekseninde day, Y ekseninde total_bill olan bir kutu grafiği oluştur.

sns.boxplot(x = "day", y = "total_bill", hue = "smoker", data = df)
plt.title("Day and Total Bill")
plt.show()
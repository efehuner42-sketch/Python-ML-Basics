import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1-Seaborn kütüphanesinden Titanic veri setini yükle:

df = sns.load_dataset('titanic')

# 2-Veri setindeki sadece sayısal (numeric) sütunları seç. (İpucu: Korelasyon sadece sayılar arasında hesaplanabilir.
digit_df = df.select_dtypes(include = ['float64', 'int64'])

# 3-Seçtiğin bu sayısal verilerin korelasyon matrisini hesapla

corr = digit_df.corr()

# 4-Çıkan bu matrisi Seaborn ile görselleştir

sns.heatmap(corr, cmap = "RdBu", square =True, annot = True, vmin = -1, vmax = 1, annot_kws = {'fontsize' : 11, 'fontweight' : 'bold'})

plt.show()

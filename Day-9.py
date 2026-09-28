import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# ==========================================
# 2. CREATE A MULTI-VARIABLE DUMMY DATASET
# ==========================================
df = pd.DataFrame({
    'Metrekare': [80, 90, 100, 120, 150, 180],
    'Oda_Sayisi': [2, 3, 3, 4, 4, 5],
    'Bina_Yasi': [10, 8, 5, 2, 1, 0],
    'Fiyat': [200000, 230000, 280000, 350000, 420000, 500000]
})

# ==========================================
# 3. DEFINE X AND y
# ==========================================
X = df.drop(["Fiyat"], axis=1).values
y = df["Fiyat"].values

# ==========================================
# 3. SPLIT THE DATA (Train & Test Split)
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# ==========================================
# 4. TRAIN THE MODEL (.fit)
# ==========================================
model = LinearRegression()
model.fit(X_train, y_train)

# ==========================================
# 5. MAKE A NEW PREDICTION (.predict)
# ==========================================
print("Predicted Price:", model.predict([[110, 3, 4]]))

# ==========================================
# 6. INSPECT COEFFICIENTS (WEIGHTS)
# ==========================================
print("Feature Coefficients:", model.coef_)
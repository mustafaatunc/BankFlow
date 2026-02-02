import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping



columns = ['checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
           'savings_account', 'employment', 'installment_rate', 'status_sex', 'guarantors',
           'residence_since', 'property', 'age', 'other_installments', 'housing',
           'existing_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'risk']

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
df = pd.read_csv(url, sep=' ', names=columns)

# 1(Good)->0(Güvenilir), 2(Bad)->1(Riskli)
df['risk'] = df['risk'].map({1: 0, 2: 1})

X = df.drop('risk', axis=1)
y = df['risk']

categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore',sparse_output=False), categorical_cols)
    ])


# Veriyi Ayırma (%80 Eğitim, %20 Test)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)
print(f"✅ Giriş boyutu: {X_train.shape[1]}")

# Sınıf Ağırlıkları
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

#  MODEL MİMARİSİ

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

input_dim = X_train.shape[1]
model = build_model(input_dim)

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    class_weight=class_weights_dict,
    callbacks=[early_stopping],
    verbose=1
)


# Grafik Çizimi
plt.figure(figsize=(14, 6))

# Doğruluk Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Test (Validation) Doğruluğu')
plt.title('Model Doğruluk Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.grid(True)

# Kayıp Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Test (Validation) Kaybı')
plt.title('Model Kayıp (Loss) Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
plt.grid(True)
plt.show()

# Karmaşıklık Matrisi
y_pred = model.predict(X_test) #(Her müşteri için 0.0 İLE 1.0 ARASINDA RİSK PUANI ÜRETİR)
y_pred_classes = (y_pred > 0.5).astype("int32") #RİSK>0.5 RİSKLİ DEĞİLSE GÜVENİLİR ŞEKLİNDE KARAR VERİR

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Güvenilir (0)', 'Riskli (1)'],
            yticklabels=['Güvenilir (0)', 'Riskli (1)'])
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Durum')
plt.title('Karmaşıklık Matrisi')
plt.show()

# Metrik Raporu
print("\n--- DETAYLI SINIFLANDIRMA RAPORU ---")
print(classification_report(y_test, y_pred_classes, target_names=['Güvenilir', 'Riskli']))

# --- main.py dosyasının EN ALTINA ekleyin ve çalıştırın ---

import joblib

# 1. Eğitilmiş Modeli Kaydet
model.save('kredi_risk_modeli.keras')
print("✅ Model başarıyla kaydedildi: kredi_risk_modeli.keras")

# 2. Ön İşleyiciyi (Scaler ve Encoder) Kaydet
# Yeni gelen ham veriyi, modelin anladığı dile çevirmek için buna mecburuz.
joblib.dump(preprocessor, 'veri_isleyici.pkl')
print("✅ Veri işleyici başarıyla kaydedildi: veri_isleyici.pkl")
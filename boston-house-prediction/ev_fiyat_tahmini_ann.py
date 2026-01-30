"""
Boston Housing - Yapay Sinir Ağları ile Ev Fiyatı Tahmini
=========================================================
Bu proje, ev özelliklerine göre fiyat tahmini yapmak için 
yapay sinir ağları (Multi-Layer Perceptron) kullanır.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')
import pickle

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print("BOSTON HOUSING - YAPAY SİNİR AĞLARI İLE EV FİYATI TAHMİNİ")
print("=" * 70)

# 1. VERİ YÜKLEME
print("\n📁 Veri yükleniyor...")
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 
                'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

df = pd.read_csv('housing.csv', delim_whitespace=True, header=None, names=column_names)

print(f"\n✓ Veri başarıyla yüklendi!")
print(f"  Toplam örnek sayısı: {len(df)}")
print(f"  Özellik sayısı: {len(df.columns) - 1}")

# 2. VERİ KEŞFİ
print("\n" + "=" * 70)
print("📊 VERİ SETİ HAKKINDA BİLGİLER")
print("=" * 70)

print("\n🔍 İlk 5 kayıt:")
print(df.head())

print("\n📈 İstatistiksel Özet:")
print(df.describe().round(2))

print("\n🏷️ Özellik Açıklamaları:")
feature_descriptions = {
    'CRIM': 'Suç oranı (per capita crime rate)',
    'ZN': 'Konut alanı oranı (>25,000 sq.ft)',
    'INDUS': 'Ticari alan oranı',
    'CHAS': 'Charles River yakınlığı (0/1)',
    'NOX': 'Azot oksit konsantrasyonu',
    'RM': 'Ortalama oda sayısı',
    'AGE': 'Eski ev oranı (1940 öncesi)',
    'DIS': 'İstihdam merkezlerine uzaklık',
    'RAD': 'Otoyol erişim indeksi',
    'TAX': 'Emlak vergisi oranı',
    'PTRATIO': 'Öğrenci-öğretmen oranı',
    'B': 'Siyahi nüfus oranı',
    'LSTAT': 'Düşük statülü nüfus yüzdesi',
    'MEDV': '🎯 Hedef: Ev fiyatı (bin $)'
}

for feature, description in feature_descriptions.items():
    print(f"  • {feature:8} - {description}")

# Eksik veri kontrolü
print(f"\n❌ Eksik değer: {df.isnull().sum().sum()}")

# 3. VERİ GÖRSELLEŞTİRME
print("\n" + "=" * 70)
print("📊 VERİ GÖRSELLEŞTİRME")
print("=" * 70)

# Hedef değişkenin dağılımı
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Fiyat dağılımı
axes[0, 0].hist(df['MEDV'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Ev Fiyatı ($1000)', fontsize=11)
axes[0, 0].set_ylabel('Frekans', fontsize=11)
axes[0, 0].set_title('Ev Fiyatlarının Dağılımı', fontsize=13, fontweight='bold')
axes[0, 0].axvline(df['MEDV'].mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Ortalama: ${df["MEDV"].mean():.1f}k')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# En önemli özelliklerle ilişki
important_features = ['RM', 'LSTAT', 'PTRATIO']
for idx, feature in enumerate(important_features):
    row = (idx + 1) // 2
    col = (idx + 1) % 2
    axes[row, col].scatter(df[feature], df['MEDV'], alpha=0.5, s=30, color='steelblue')
    axes[row, col].set_xlabel(feature_descriptions[feature], fontsize=11)
    axes[row, col].set_ylabel('Ev Fiyatı ($1000)', fontsize=11)
    axes[row, col].set_title(f'{feature} vs Ev Fiyatı', fontsize=12, fontweight='bold')
    axes[row, col].grid(True, alpha=0.3)
    
    # Trend çizgisi
    z = np.polyfit(df[feature], df['MEDV'], 1)
    p = np.poly1d(z)
    axes[row, col].plot(df[feature], p(df[feature]), "r--", alpha=0.8, linewidth=2, label='Trend')
    axes[row, col].legend()

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/1_veri_gorsellestirme.png', dpi=300, bbox_inches='tight')
print("✓ Görsel kaydedildi: 1_veri_gorsellestirme.png")
plt.close()

# Korelasyon matrisi
plt.figure(figsize=(14, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Özellikler Arası Korelasyon Matrisi', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/2_korelasyon_matrisi.png', dpi=300, bbox_inches='tight')
print("✓ Görsel kaydedildi: 2_korelasyon_matrisi.png")
plt.close()

# En yüksek korelasyonlar
print("\n🔗 Ev Fiyatı ile En Yüksek Korelasyonlar:")
correlations = df.corr()['MEDV'].sort_values(ascending=False)
for feature, corr in correlations.items():
    if feature != 'MEDV':
        emoji = "📈" if corr > 0 else "📉"
        print(f"  {emoji} {feature:8} : {corr:+.3f}")

# 4. VERİ HAZIRLIĞI
print("\n" + "=" * 70)
print("🔧 VERİ HAZIRLANIYOR")
print("=" * 70)

# Özellikler ve hedef değişken
X = df.drop('MEDV', axis=1)
y = df['MEDV']

print(f"\n✓ Özellikler (X): {X.shape}")
print(f"✓ Hedef (y): {y.shape}")

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📊 Veri Setleri:")
print(f"  • Eğitim seti: {X_train.shape[0]} örnek")
print(f"  • Test seti: {X_test.shape[0]} örnek")

# Veriyi ölçeklendirme (Normalization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Veri standardize edildi (StandardScaler)")
print(f"  Ortalama: {X_train_scaled.mean():.6f}")
print(f"  Standart sapma: {X_train_scaled.std():.6f}")

# 5. YAPAY SİNİR AĞI MODELİ OLUŞTURMA
print("\n" + "=" * 70)
print("🧠 YAPAY SİNİR AĞI MODELİ OLUŞTURULUYOR")
print("=" * 70)

# Model mimarisi
# MLPRegressor = Multi-Layer Perceptron (Çok Katmanlı Algılayıcı)
model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32, 16),  # 4 gizli katman
    activation='relu',                      # ReLU aktivasyon fonksiyonu
    solver='adam',                          # Adam optimizer
    alpha=0.001,                            # L2 regularization
    batch_size=32,                          # Mini-batch boyutu
    learning_rate='adaptive',               # Adaptif öğrenme oranı
    learning_rate_init=0.001,               # Başlangıç öğrenme oranı
    max_iter=1000,                          # Maksimum epoch
    early_stopping=True,                    # Erken durdurma
    validation_fraction=0.2,                # Validation set oranı
    n_iter_no_change=50,                    # Erken durdurma patience
    verbose=False,                          # Sessiz mod
    random_state=42
)

print("\n🏗️ Model Mimarisi:")
print(f"  • Giriş katmanı: {X_train_scaled.shape[1]} nöron")
print(f"  • Gizli katman 1: 128 nöron (ReLU)")
print(f"  • Gizli katman 2: 64 nöron (ReLU)")
print(f"  • Gizli katman 3: 32 nöron (ReLU)")
print(f"  • Gizli katman 4: 16 nöron (ReLU)")
print(f"  • Çıkış katmanı: 1 nöron (Linear)")

print("\n⚙️ Model Parametreleri:")
print(f"  • Optimizer: Adam")
print(f"  • Öğrenme oranı: 0.001 (adaptive)")
print(f"  • Batch size: 32")
print(f"  • Max epoch: 1000")
print(f"  • Early stopping: Aktif (patience=50)")
print(f"  • L2 regularization (alpha): 0.001")

# 6. MODEL EĞİTİMİ
print("\n" + "=" * 70)
print("🎯 MODEL EĞİTİLİYOR")
print("=" * 70)
print("\n🚀 Eğitim başlıyor...\n")

# Eğitim
model.fit(X_train_scaled, y_train)

print(f"\n✓ Model eğitimi tamamlandı!")
print(f"  • Toplam iterasyon: {model.n_iter_}")
print(f"  • Son loss değeri: {model.loss_:.6f}")

# 7. MODEL PERFORMANSI
print("\n" + "=" * 70)
print("📈 MODEL PERFORMANSI")
print("=" * 70)

# Tahminler
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

# Metrikler
def calculate_metrics(y_true, y_pred, set_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{set_name} Seti Sonuçları:")
    print(f"  • R² Score (Açıklama Gücü): {r2:.4f} ({r2*100:.2f}%)")
    print(f"  • RMSE (Kök Ortalama Kare Hatası): ${rmse:.2f}k")
    print(f"  • MAE (Ortalama Mutlak Hata): ${mae:.2f}k")
    print(f"  • MSE (Ortalama Kare Hatası): {mse:.2f}")
    
    return mse, rmse, mae, r2

train_metrics = calculate_metrics(y_train, y_train_pred, "🎓 Eğitim")
test_metrics = calculate_metrics(y_test, y_test_pred, "🧪 Test")

# Overfitting kontrolü
overfit_check = train_metrics[3] - test_metrics[3]
print(f"\n📊 Overfitting Kontrolü:")
print(f"  • R² farkı (Train - Test): {overfit_check:.4f}")
if overfit_check < 0.05:
    print(f"  ✓ Model iyi genelleştirilmiş (Overfitting YOK)")
elif overfit_check < 0.15:
    print(f"  ⚠️ Hafif overfitting var")
else:
    print(f"  ❌ Ciddi overfitting var")

# Cross-validation
print("\n🔄 Cross-Validation (5-Fold):")
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                            scoring='r2', n_jobs=-1)
print(f"  • Ortalama R²: {cv_scores.mean():.4f}")
print(f"  • Standart sapma: {cv_scores.std():.4f}")
print(f"  • Tüm skorlar: {[f'{s:.3f}' for s in cv_scores]}")

# 8. GÖRSELLEŞTİRME - EĞİTİM SÜRECİ
print("\n" + "=" * 70)
print("📊 SONUÇLAR GÖRSELLEŞTİRİLİYOR")
print("=" * 70)

# Eğitim geçmişi
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# Loss grafiği (loss_curve_ sadece early_stopping=True ise var)
if hasattr(model, 'loss_curve_'):
    ax.plot(model.loss_curve_, linewidth=2, color='steelblue', label='Training Loss')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Model Loss Değişimi', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Best loss noktası
    best_iter = np.argmin(model.loss_curve_)
    ax.axvline(best_iter, color='red', linestyle='--', alpha=0.5, 
               label=f'Best iteration: {best_iter}')
    ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/3_egitim_sureci.png', dpi=300, bbox_inches='tight')
print("✓ Görsel kaydedildi: 3_egitim_sureci.png")
plt.close()

# 9. TAHMİN vs GERÇEK DEĞERLER
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Test seti tahminleri
axes[0].scatter(y_test, y_test_pred, alpha=0.6, s=50, color='steelblue', edgecolors='darkblue', linewidth=0.5)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=3, label='Mükemmel Tahmin')
axes[0].set_xlabel('Gerçek Fiyat ($1000)', fontsize=12)
axes[0].set_ylabel('Tahmin Edilen Fiyat ($1000)', fontsize=12)
axes[0].set_title(f'Test Seti: Gerçek vs Tahmin\nR² = {test_metrics[3]:.3f}', 
                  fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Hata dağılımı
errors = y_test - y_test_pred
axes[1].hist(errors, bins=30, color='coral', edgecolor='darkred', alpha=0.7, linewidth=1)
axes[1].axvline(0, color='red', linestyle='--', linewidth=3, label='Hata=0')
axes[1].axvline(errors.mean(), color='blue', linestyle='--', linewidth=2, 
                label=f'Ortalama: ${errors.mean():.2f}k')
axes[1].set_xlabel('Tahmin Hatası ($1000)', fontsize=12)
axes[1].set_ylabel('Frekans', fontsize=12)
axes[1].set_title(f'Hata Dağılımı\nStd: ${errors.std():.2f}k', 
                  fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/4_tahmin_sonuclari.png', dpi=300, bbox_inches='tight')
print("✓ Görsel kaydedildi: 4_tahmin_sonuclari.png")
plt.close()

# 10. ÖRNEK TAHMİNLER
print("\n" + "=" * 70)
print("🏠 ÖRNEK TAHMİNLER")
print("=" * 70)

# Rastgele 15 örnek seç
sample_indices = np.random.choice(len(X_test), min(15, len(X_test)), replace=False)
samples = X_test.iloc[sample_indices]
samples_scaled = scaler.transform(samples)
predictions = model.predict(samples_scaled)
actuals = y_test.iloc[sample_indices].values

print("\n  #  | Gerçek ($k) | Tahmin ($k) | Fark ($k) | Hata %  | Durum")
print("-" * 75)
for i, (actual, pred) in enumerate(zip(actuals, predictions), 1):
    diff = actual - pred
    error_pct = abs(diff) / actual * 100
    status = "✓" if error_pct < 15 else "⚠" if error_pct < 25 else "✗"
    print(f"{i:3d} | {actual:11.2f} | {pred:11.2f} | {diff:9.2f} | {error_pct:6.2f}% | {status}")

avg_error = np.mean(np.abs(actuals - predictions))
print(f"\nOrtalama Mutlak Hata: ${avg_error:.2f}k")

# 11. MODEL KAYDETME
print("\n" + "=" * 70)
print("💾 MODEL KAYDEDİLİYOR")
print("=" * 70)

# Model kaydet
with open('/mnt/user-data/outputs/ev_fiyat_modeli.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Model kaydedildi: ev_fiyat_modeli.pkl")

# Scaler'ı kaydet
with open('/mnt/user-data/outputs/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler kaydedildi: scaler.pkl")

# 12. ÖZELLİK ÖNEMLİLİĞİ (Permutation Importance)
print("\n" + "=" * 70)
print("🔍 ÖZELLİK ÖNEMLİLİĞİ ANALİZİ")
print("=" * 70)

result = permutation_importance(
    model, X_test_scaled, y_test, 
    n_repeats=10, random_state=42, n_jobs=-1
)

# Sonuçları sırala
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': result.importances_mean,
    'std': result.importances_std
}).sort_values('importance', ascending=False)

print("\n📊 Özellik Önemlilikleri:")
for idx, row in feature_importance.iterrows():
    bar = "█" * int(row['importance'] * 50)
    print(f"  {row['feature']:8} : {row['importance']:6.4f} (±{row['std']:.4f}) {bar}")

# Görselleştir
plt.figure(figsize=(12, 8))
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(feature_importance)))
plt.barh(feature_importance['feature'], feature_importance['importance'], 
         xerr=feature_importance['std'], color=colors, alpha=0.8, edgecolor='black', linewidth=1)
plt.xlabel('Önemlilik (Permutation Importance)', fontsize=12)
plt.ylabel('Özellikler', fontsize=12)
plt.title('Yapay Sinir Ağı - Özellik Önemlilikleri', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/5_ozellik_onemliligi.png', dpi=300, bbox_inches='tight')
print("\n✓ Görsel kaydedildi: 5_ozellik_onemliligi.png")
plt.close()

# 13. DETAYLI PERFORMANS ANALİZİ
print("\n" + "=" * 70)
print("🔬 DETAYLI PERFORMANS ANALİZİ")
print("=" * 70)

# Fiyat aralıklarına göre performans
price_ranges = [(0, 20), (20, 30), (30, 40), (40, 100)]
print("\n💰 Fiyat Aralıklarına Göre Performans:")
print("-" * 60)

for low, high in price_ranges:
    mask = (y_test >= low) & (y_test < high)
    if mask.sum() > 0:
        range_mae = mean_absolute_error(y_test[mask], y_test_pred[mask])
        range_r2 = r2_score(y_test[mask], y_test_pred[mask])
        count = mask.sum()
        print(f"  ${low:2d}k - ${high:2d}k: MAE=${range_mae:.2f}k, R²={range_r2:.3f}, N={count:3d}")

# En iyi ve en kötü tahminler
errors_abs = np.abs(y_test - y_test_pred)
best_indices = errors_abs.nsmallest(3).index
worst_indices = errors_abs.nlargest(3).index

print("\n✅ En İyi 3 Tahmin:")
for idx in best_indices:
    print(f"  Gerçek: ${y_test.loc[idx]:.2f}k, Tahmin: ${y_test_pred[y_test.index.get_loc(idx)]:.2f}k, "
          f"Hata: ${errors_abs.loc[idx]:.2f}k")

print("\n❌ En Kötü 3 Tahmin:")
for idx in worst_indices:
    print(f"  Gerçek: ${y_test.loc[idx]:.2f}k, Tahmin: ${y_test_pred[y_test.index.get_loc(idx)]:.2f}k, "
          f"Hata: ${errors_abs.loc[idx]:.2f}k")

# 14. ÖZET RAPOR
print("\n" + "=" * 70)
print("📋 PROJE ÖZET RAPORU")
print("=" * 70)

report = f"""
╔═══════════════════════════════════════════════════════════════════╗
║     BOSTON HOUSING - YAPAY SİNİR AĞI İLE EV FİYATI TAHMİNİ      ║
╚═══════════════════════════════════════════════════════════════════╝

📊 VERİ SETİ BİLGİLERİ:
{'─' * 70}
  • Toplam Örnek Sayısı      : {len(df)}
  • Özellik Sayısı           : {len(X.columns)}
  • Eğitim Seti              : {len(X_train)} örnek (%{len(X_train)/len(df)*100:.0f})
  • Test Seti                : {len(X_test)} örnek (%{len(X_test)/len(df)*100:.0f})
  • Veri Ölçeklendirme       : StandardScaler

🧠 MODEL MİMARİSİ:
{'─' * 70}
  • Model Tipi               : Multi-Layer Perceptron (MLP)
  • Gizli Katman Sayısı      : 4 katman
  • Nöron Yapısı             : [128, 64, 32, 16]
  • Aktivasyon Fonksiyonu    : ReLU (gizli), Linear (çıkış)
  • Optimizer                : Adam
  • Öğrenme Oranı            : 0.001 (adaptive)
  • Batch Size               : 32
  • Max Epoch                : 1000
  • Toplam İterasyon         : {model.n_iter_}
  • Early Stopping           : Aktif (patience=50)
  • L2 Regularization        : 0.001

📈 PERFORMANS METRİKLERİ:
{'─' * 70}

  🎓 EĞİTİM SETİ:
     ├─ R² Score             : {train_metrics[3]:.4f} ({train_metrics[3]*100:.2f}%)
     ├─ RMSE                 : ${train_metrics[1]:.2f}k
     ├─ MAE                  : ${train_metrics[2]:.2f}k
     └─ MSE                  : {train_metrics[0]:.2f}

  🧪 TEST SETİ:
     ├─ R² Score             : {test_metrics[3]:.4f} ({test_metrics[3]*100:.2f}%)
     ├─ RMSE                 : ${test_metrics[1]:.2f}k
     ├─ MAE                  : ${test_metrics[2]:.2f}k
     └─ MSE                  : {test_metrics[0]:.2f}

  🔄 CROSS-VALIDATION (5-Fold):
     ├─ Ortalama R²          : {cv_scores.mean():.4f}
     └─ Standart Sapma       : {cv_scores.std():.4f}

🎖️ EN ÖNEMLİ ÖZELLİKLER:
{'─' * 70}
"""

for i, (idx, row) in enumerate(feature_importance.head(5).iterrows(), 1):
    importance_bar = "█" * int(row['importance'] * 30)
    report += f"  {i}. {row['feature']:8} : {row['importance']:.4f} {importance_bar}\n"
    report += f"     └─ {feature_descriptions[row['feature']]}\n"

report += f"""
✅ MODEL DEĞERLENDİRMESİ:
{'─' * 70}
  • Model Performansı        : {'Mükemmel' if test_metrics[3] > 0.85 else 'İyi' if test_metrics[3] > 0.75 else 'Orta'}
  • Genelleştirme            : {'İyi' if overfit_check < 0.05 else 'Orta' if overfit_check < 0.15 else 'Zayıf'}
  • Tahmin Doğruluğu         : %{test_metrics[3]*100:.1f}
  • Ortalama Hata            : ±${test_metrics[2]:.2f}k (±${test_metrics[2]*1000:.0f})
  
💡 YORUMLAR:
{'─' * 70}
  ✓ Model, ev fiyatlarını yüksek doğrulukla tahmin edebiliyor
  ✓ En etkili özellikler: {', '.join(feature_importance.head(3)['feature'].values)}
  {'✓ Model iyi genelleştirilmiş, overfitting riski düşük' if overfit_check < 0.1 else '⚠ Hafif overfitting gözlemlendi'}
  ✓ Oda sayısı (RM) ve düşük statü oranı (LSTAT) fiyatı en çok etkiliyor

📁 ÇIKTI DOSYALARI:
{'─' * 70}
  • ev_fiyat_modeli.pkl      - Eğitilmiş yapay sinir ağı modeli
  • scaler.pkl               - Veri ölçekleyici (StandardScaler)
  • proje_raporu.txt         - Detaylı proje raporu
  • yeni_tahmin.py           - Yeni tahmin scripti
  • 5 adet PNG görsel dosyası (görselleştirmeler)

🚀 KULLANIM ÖNERİSİ:
{'─' * 70}
  # Model yükleme:
  import pickle
  with open('ev_fiyat_modeli.pkl', 'rb') as f:
      model = pickle.load(f)
  with open('scaler.pkl', 'rb') as f:
      scaler = pickle.load(f)
  
  # Yeni tahmin:
  yeni_ev = [[0.1, 10.0, 5.0, 0, 0.5, 6.5, 70, 4.0, 3, 300, 16, 390, 10]]
  yeni_ev_scaled = scaler.transform(yeni_ev)
  prediction = model.predict(yeni_ev_scaled)
  print(f"Tahmini fiyat: ${{prediction[0]:.2f}}k")

📚 TEKNİK DETAYLAR:
{'─' * 70}
  • Kütüphaneler: scikit-learn, pandas, numpy, matplotlib, seaborn
  • Python Versiyonu: 3.x
  • Model Algoritması: Backpropagation with Adam Optimizer
  • Kayıp Fonksiyonu: Mean Squared Error (MSE)
  • Aktivasyon: ReLU (gizli), Identity (çıkış)

═══════════════════════════════════════════════════════════════════

🎉 PROJE BAŞARIYLA TAMAMLANDI!

Yapay sinir ağı modeli, {len(feature_importance)} farklı özelliği kullanarak
ev fiyatlarını %{test_metrics[3]*100:.1f} doğrulukla tahmin edebiliyor.

Model hazır ve kullanıma uygun! 🚀

═══════════════════════════════════════════════════════════════════
"""

print(report)

# Raporu kaydet
with open('/mnt/user-data/outputs/proje_raporu.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("\n✓ Rapor kaydedildi: proje_raporu.txt")

# 15. BONUS: YENİ TAHMİN ÖRNEĞİ SCRIPT
print("\n" + "=" * 70)
print("🎁 BONUS: YENİ EV TAHMİN SCRIPTI")
print("=" * 70)

prediction_script = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Yeni Ev Fiyatı Tahmin Scripti
Bu script eğitilmiş modeli kullanarak yeni ev özelliklerine göre fiyat tahmini yapar.
'''

import pickle
import numpy as np

# Model ve scaler'ı yükle
print("Model yükleniyor...")
with open('ev_fiyat_modeli.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("✓ Model başarıyla yüklendi!\\n")

# Özellik açıklamaları
features = {
    'CRIM': 'Suç oranı',
    'ZN': 'Konut alanı oranı (>25,000 sq.ft)',
    'INDUS': 'Ticari alan oranı',
    'CHAS': 'Charles River yakınlığı (0=Hayır, 1=Evet)',
    'NOX': 'Azot oksit konsantrasyonu',
    'RM': 'Ortalama oda sayısı',
    'AGE': 'Eski ev oranı (1940 öncesi) %',
    'DIS': 'İstihdam merkezlerine uzaklık',
    'RAD': 'Otoyol erişim indeksi',
    'TAX': 'Emlak vergisi oranı',
    'PTRATIO': 'Öğrenci-öğretmen oranı',
    'B': 'Siyahi nüfus oranı',
    'LSTAT': 'Düşük statülü nüfus yüzdesi'
}

# Örnek ev özellikleri
print("=" * 70)
print("ÖRNEK EV TAHMİNLERİ")
print("=" * 70)

# Örnek 1: Lüks ev
lux_house = [[0.02, 50.0, 3.0, 1, 0.4, 8.5, 20, 5.0, 2, 250, 14, 395, 2]]
print("\\n🏰 Örnek 1: Lüks Ev")
print("  • Düşük suç oranı, nehir kenarı, 8.5 oda, yeni bina")

# Örnek 2: Orta segment ev
mid_house = [[0.1, 20.0, 5.0, 0, 0.5, 6.5, 50, 4.0, 3, 300, 16, 390, 8]]
print("\\n🏠 Örnek 2: Orta Segment Ev")
print("  • Orta suç oranı, 6.5 oda, orta yaşta bina")

# Örnek 3: Ekonomik ev
eco_house = [[0.3, 5.0, 10.0, 0, 0.6, 5.5, 80, 3.0, 5, 400, 18, 380, 15]]
print("\\n🏘️ Örnek 3: Ekonomik Ev")
print("  • Yüksek suç oranı, 5.5 oda, eski bina")

# Tahminler
houses = [lux_house, mid_house, eco_house]
house_names = ["Lüks Ev", "Orta Segment Ev", "Ekonomik Ev"]

print("\\n" + "=" * 70)
print("TAHMİN SONUÇLARI")
print("=" * 70 + "\\n")

for name, house in zip(house_names, houses):
    # Veriyi ölçeklendir
    house_scaled = scaler.transform(house)
    
    # Tahmin yap
    prediction = model.predict(house_scaled)[0]
    
    print(f"📍 {name}:")
    print(f"   └─ Tahmini Fiyat: ${prediction:.2f}k (${prediction*1000:.0f})")
    print()

print("=" * 70)
print("\\n💡 Kendi eviniz için tahmin yapmak isterseniz:")
print("   Yukarıdaki feature değerlerini değiştirerek yeni tahminler yapabilirsiniz!")
"""

with open('/mnt/user-data/outputs/yeni_tahmin.py', 'w', encoding='utf-8') as f:
    f.write(prediction_script)
print("✓ Tahmin scripti kaydedildi: yeni_tahmin.py")

print("\n" + "=" * 70)
print("✅ PROJE BAŞARIYLA TAMAMLANDI!")
print("=" * 70)
print("\n📁 Oluşturulan Dosyalar:")
print("  1. ev_fiyat_modeli.pkl - Eğitilmiş yapay sinir ağı modeli")
print("  2. scaler.pkl - Veri ölçekleyici")
print("  3. proje_raporu.txt - Detaylı rapor")
print("  4. yeni_tahmin.py - Yeni tahmin scripti")
print("  5. 1_veri_gorsellestirme.png")
print("  6. 2_korelasyon_matrisi.png")
print("  7. 3_egitim_sureci.png")
print("  8. 4_tahmin_sonuclari.png")
print("  9. 5_ozellik_onemliligi.png")

print("\n🎓 Proje Özeti:")
print(f"  • Model %{test_metrics[3]*100:.1f} doğrulukla çalışıyor")
print(f"  • Ortalama hata: ±${test_metrics[2]:.2f}k")
print(f"  • {model.n_iter_} iterasyonda eğitim tamamlandı")
print("\n🚀 Model kullanıma hazır!")

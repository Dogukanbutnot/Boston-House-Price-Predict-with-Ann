#!/usr/bin/env python3
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

print("✓ Model başarıyla yüklendi!\n")

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
print("\n🏰 Örnek 1: Lüks Ev")
print("  • Düşük suç oranı, nehir kenarı, 8.5 oda, yeni bina")

# Örnek 2: Orta segment ev
mid_house = [[0.1, 20.0, 5.0, 0, 0.5, 6.5, 50, 4.0, 3, 300, 16, 390, 8]]
print("\n🏠 Örnek 2: Orta Segment Ev")
print("  • Orta suç oranı, 6.5 oda, orta yaşta bina")

# Örnek 3: Ekonomik ev
eco_house = [[0.3, 5.0, 10.0, 0, 0.6, 5.5, 80, 3.0, 5, 400, 18, 380, 15]]
print("\n🏘️ Örnek 3: Ekonomik Ev")
print("  • Yüksek suç oranı, 5.5 oda, eski bina")

# Tahminler
houses = [lux_house, mid_house, eco_house]
house_names = ["Lüks Ev", "Orta Segment Ev", "Ekonomik Ev"]

print("\n" + "=" * 70)
print("TAHMİN SONUÇLARI")
print("=" * 70 + "\n")

for name, house in zip(house_names, houses):
    # Veriyi ölçeklendir
    house_scaled = scaler.transform(house)
    
    # Tahmin yap
    prediction = model.predict(house_scaled)[0]
    
    print(f"📍 {name}:")
    print(f"   └─ Tahmini Fiyat: ${prediction:.2f}k (${prediction*1000:.0f})")
    print()

print("=" * 70)
print("\n💡 Kendi eviniz için tahmin yapmak isterseniz:")
print("   Yukarıdaki feature değerlerini değiştirerek yeni tahminler yapabilirsiniz!")

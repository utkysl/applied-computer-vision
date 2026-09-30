import matplotlib.pyplot as plt
import numpy as np 
import cv2
import os

# Scriptin bulunduğu dizini temel alarak assets klasörünün tam yolunu oluştur
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, '..', 'assets', 'sample.jpg')

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Görsel bulunamadı! Aranan yol: {image_path}")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"Görüntü Boyutları (H, W, C): {img.shape}")

# 2. Kanalları dilimle (Slicing)
R = img[:, :, 0]
G = img[:, :, 1]
B = img[:, :, 2]

# 3. ITU-R BT.601 Lüminesans denklemiyle griye çevir
gray_custom = 0.299 * R + 0.587 * G + 0.114 * B
gray_custom = np.uint8(gray_custom)

# 4. OpenCV'nin yerleşik fonksiyonuyla karşılaştır
gray_cv = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 5. İki yöntem arasındaki maksimum piksel farkını ölç
diff = np.max(np.abs(gray_custom.astype(int) - gray_cv.astype(int)))
print(f"Özel Matris Hesabı ile OpenCV Arasındaki Maksimum Piksel Farkı: {diff}")

# 6. Sonuçları görselleştir ve kaydet
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

axes[0].imshow(img)
axes[0].set_title("Orijinal (RGB)")

axes[1].imshow(R, cmap='Reds')
axes[1].set_title("Kırmızı Kanalı (R)")

axes[2].imshow(G, cmap='Greens')
axes[2].set_title("Yeşil Kanalı (G)")

axes[3].imshow(B, cmap='Blues')
axes[3].set_title("Mavi Kanalı (B)")

axes[4].imshow(gray_custom, cmap='gray')
axes[4].set_title(f"Özel Gri Ton (Fark: {diff})")

for ax in axes:
    ax.axis('off')

# results klasörüne kaydet
results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)
save_path = os.path.join(results_dir, 'channel_analysis.png')

plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"Analiz görseli kaydedildi: {save_path}")
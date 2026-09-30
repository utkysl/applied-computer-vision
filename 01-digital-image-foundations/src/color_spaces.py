import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1. Görseli dinamik yol kullanarak yükle
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, '..', 'assets', 'sample.jpg')

img_bgr = cv2.imread(image_path)
if img_bgr is None:
    raise FileNotFoundError(f"Görsel bulunamadı! Aranan yol: {image_path}")

# Matplotlib için RGB, renk analizi için HSV uzayına çevir
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# 2. HSV Kanallarını ayrıştır
# H: 0-179 (Renk Özü), S: 0-255 (Doygunluk), V: 0-255 (Parlaklık)
H = img_hsv[:, :, 0]
S = img_hsv[:, :, 1]
V = img_hsv[:, :, 2]

# 3. Renk Segmentasyonu / Eşikleme (Örnek: Mavi tonları izolasyonu)
# İhtiyacına göre bu aralıkları değiştirebilirsin:
# Mavi: [100, 50, 50] - [130, 255, 255]
# Kırmızı (OpenCV'de ikiye bölünür): [0, 100, 100] - [10, 255, 255]
lower_blue = np.array([100, 50, 50], dtype=np.uint8)
upper_blue = np.array([130, 255, 255], dtype=np.uint8)

# Maske: Belirlenen aralıktaki pikseller 255 (beyaz), diğerleri 0 (siyah)
mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

# Bitwise AND: Orijinal görüntü ile maskeyi çarparak sadece hedef rengi koru
segmented = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

# 4. Görselleştirme
fig, axes = plt.subplots(1, 5, figsize=(22, 5))

axes[0].imshow(img_rgb)
axes[0].set_title("Orijinal (RGB)")

axes[1].imshow(H, cmap='hsv')
axes[1].set_title("Hue (Renk Özü)")

axes[2].imshow(S, cmap='gray')
axes[2].set_title("Saturation (Doygunluk)")

axes[3].imshow(V, cmap='gray')
axes[3].set_title("Value (Parlaklık)")

axes[4].imshow(segmented)
axes[4].set_title("Segmentasyon (İzole Renk)")

for ax in axes:
    ax.axis('off')

# 5. Sonuçları kaydet
results_dir = os.path.join(script_dir, '..', 'results')
os.makedirs(results_dir, exist_ok=True)
save_path = os.path.join(results_dir, 'color_space_analysis.png')

plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Renk uzayı analiz grafiği kaydedildi: {save_path}")
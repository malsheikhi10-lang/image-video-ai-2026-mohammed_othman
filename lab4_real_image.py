from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1. فتح الصورة الملونة الأصلية
image_path = r"C:\Users\PC-LAB1\Desktop\image_processing_course\222.jpg"
img = Image.open(image_path)
image = np.array(img)

# التأكد من أن الصورة ملونة بثلاث قنوات (RGB)
if len(image.shape) == 2:
    image = np.stack((image,)*3, axis=-1)
elif image.shape[2] == 4:
    image = image[:, :, :3]

# 2. استخراج الـ LUT الموحد للحفاظ على ألوان الصورة الطبيعية من خلال النسخة الرمادية
gray_image = np.array(Image.fromarray(image).convert("L"))

hist_orig = np.zeros(256, dtype=int)
for i in range(gray_image.shape[0]):
    for j in range(gray_image.shape[1]):
        hist_orig[gray_image[i, j]] += 1

cdf = np.zeros(256, dtype=int)
cdf[0] = hist_orig[0]
for i in range(1, 256):
    cdf[i] = cdf[i-1] + hist_orig[i]

non_zero_indices = np.nonzero(cdf)[0]
cdf_min = cdf[non_zero_indices[0]] if len(non_zero_indices) > 0 else 0
total_pixels = gray_image.shape[0] * gray_image.shape[1]

equalized_lut = np.zeros(256, dtype=np.uint8)
for i in range(256):
    if cdf[i] > 0 and total_pixels > cdf_min:
        val = ((cdf[i] - cdf_min) / (total_pixels - cdf_min)) * 255
        equalized_lut[i] = np.clip(val, 0, 255)

# 3. تطبيق جدول التحويل (LUT) على القنوات الثلاثة معا (باستخدام اللوبين)
equalized_image = np.zeros_like(image)
for c in range(3):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            equalized_image[i, j, c] = equalized_lut[image[i, j, c]]

# 4. حساب الهستغرام الملون (RGB) للصورة الأصلية
hist_rgb_orig = np.zeros((256, 3), dtype=int)
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        hist_rgb_orig[image[i, j, :], [0, 1, 2]] += [1, 1, 1]

# 5. حساب الهستغرام الملون (RGB) للصورة بعد الاكولايزيشن
hist_rgb_eq = np.zeros((256, 3), dtype=int)
for i in range(equalized_image.shape[0]):
    for j in range(equalized_image.shape[1]):
        hist_rgb_eq[equalized_image[i, j, :], [0, 1, 2]] += [1, 1, 1]

# 6. رسم النتائج في فيقر واحد مقسم 2x2
plt.figure(figsize=(12, 10))

# [1] اليسار العلوي: الصورة الأصلية
plt.subplot(2, 2, 1)
plt.imshow(image)
plt.title("1. Original RGB Image")
plt.axis("off")

# [2] اليمين العلوي: الهستغرام الأصلي
plt.subplot(2, 2, 2)
colors = ['red', 'green', 'blue']
for idx, color in enumerate(colors):
    plt.plot(hist_rgb_orig[:, idx], color=color, label=f'{color.upper()} Channel')
plt.title("2. Original RGB Histogram")
plt.xlabel("Intensity (0-255)")
plt.ylabel("Pixel Count")
plt.legend()
plt.grid(True)

# [3] اليسار السفلي: الصورة بعد الاكولايزيشن
plt.subplot(2, 2, 3)
plt.imshow(equalized_image)
plt.title("3. Equalized RGB Image")
plt.axis("off")

# [4] اليمين السفلي: الهستغرام بعد الاكولايزيشن
plt.subplot(2, 2, 4)
for idx, color in enumerate(colors):
    plt.plot(hist_rgb_eq[:, idx], color=color, label=f'{color.upper()} Channel')
plt.title("4. Equalized RGB Histogram")
plt.xlabel("Intensity (0-255)")
plt.ylabel("Pixel Count")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
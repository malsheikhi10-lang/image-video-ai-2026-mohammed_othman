import numpy as np
import matplotlib.pyplot as plt

# إنشاء صورة سوداء أبعادها 400 في 400 بكسل
image = np.zeros((400, 400), dtype=np.uint8)

# رسم مربع أبيض في المنتصف
image[100:300, 100:300] = 255

# عرض الصورة باستخدام مكتبة Matplotlib
plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()
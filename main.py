import cv2 as cv
import numpy as np
import constants as const
import img_processor as ip
import matplotlib.pyplot as plt

# INICIALIZAÇÃO
img = cv.imread("img.png", 0)
neighborhood = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype=np.uint8)

# CALCULAR THRESHOLD
img = ip.get_threshold(img)

# DILATANDO LETRAS VERTICALMENTE (AGRUPANDO ACENTUAÇÃO À LETRAS)
kernel_letter = np.array([[0, 1, 0],
                          [0, 1, 0],
                          [0, 1, 0]], dtype=np.uint8)
img_letter = ip.apply_effect(img, kernel_letter, const.DILATE)

# DILATANDO LETRAS EM TODAS AS DIREÇÕES (AGRUPANDO LETRAS DE PALAVRA)
kernel_word = np.ones((5, 5), dtype=np.uint8)
img_word = ip.apply_effect(img, kernel_word, const.DILATE)

# SEGMENTANDO LETRAS
letters = ip.segment_image(img_letter, neighborhood)

# SEGMENTANDO PALAVRAS
words = ip.segment_image(img_word, neighborhood)

# RESULTADO
plot, (f1, f2, f3) = plt.subplots(3)
plot.suptitle('Resultado')

img_letter = cv.cvtColor(img_letter, cv.COLOR_BGR2RGB)
f1.set_title('Imagem após dilatação vertical')
f1.imshow(img_letter)

img_word = cv.cvtColor(img_word, cv.COLOR_BGR2RGB)
f2.set_title('Imagem após dilatação extrema')
f2.imshow(img_word)

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = ip.draw_bounds(img, letters, (0, 255, 0))
img = ip.draw_bounds(img, words, (255, 0, 0))
f3.set_title('Imagem final')
f3.imshow(img)

plt.subplots_adjust(hspace=0.75)
plt.show()
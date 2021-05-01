import cv2 as cv
import numpy as np
import constants as const
import img_processor as ip

# INICIALIZAÇÃO
img = cv.imread("img.png", 0)
neighborhood = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype=np.uint8)

# CALCULAR THRESHOLD
ret, img = cv.threshold(img, 200, 255, cv.THRESH_BINARY)

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

# DESENHANDO RETÂNGULOS
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = ip.draw_bounds(img, letters, (0, 255, 0))
img = ip.draw_bounds(img, words, (255, 0, 0))

# RESULTADO
ip.display_img(img)
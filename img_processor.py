import cv2 as cv
import math
import matplotlib.pyplot as plt
import numpy as np
import constants as const

def check_hit(img, pos, kernel):
    """
    Verifica a vizinhança de um pixel baseado em uma matriz de congruência.
    
    Parâmetros:
    - img: Matriz de pixels.
    - pos: Posição do pixel na imagem.
    - kernel: Matriz de congruência.

    Retorno:
    - 2 (FIT) se todos os vizinhos possuem valor 0;
    - 1 (HIT) se ao menos um vizinho possui valor 0;
    - 0 (MISS) caso contrário.
    """
    
    x, y = pos
    neighbours = get_neighbours(img, (x, y), kernel)
    count = len(neighbours)

    if (count == np.count_nonzero(kernel)):
        return const.FIT

    elif (count > 0):
        return const.HIT

    else:
        return const.MISS

def apply_effect(img, kernel, mode):
    """
    Aplica o efeito de erosão ou dilatação em uma imagem.

    Parâmetros:
    - img: Matriz de pixels;
    - kernel: Matriz de Congruência;
    - mode: Tipo de efeito (EROSION = 0, DILATE = 1).

    Retorno:
    Matriz de pixels com efeito aplicado.
    """

    width, height = img.shape
    result = np.zeros((width, height), dtype=np.uint8)

    for x in range(0, width):
        for y in range(0, height):
            
            hit = check_hit(img, (x, y), kernel)

            if (hit == const.FIT or (hit == const.HIT and mode == const.DILATE)):
                result[x, y] = 0
            else:
                result[x, y] = 255
                
    return result

def histogram(img):
    """
    Calcula o histograma de uma imagem.

    Parâmetros:
    - img: Matriz de pixels.

    Retorno:
    Lista com 256 inteiros contendo a recorrência de cada valor na imagem.
    """

    width, height = img.shape
    hist = np.zeros(256)

    for x in range(0, width):
        for y in range(0, height):
            value = img[x, y]
            hist[value] = hist[value] + 1
    
    return hist

def segment_image(img, kernel):
    """
    Segmenta uma imagem em grupos de proximidade.

    Parâmetros:
    - img: Matriz de pixels;
    - kernel: Matriz de Congruência

    Retorno:
    Dictionary contendo todos os grupos encontrados. Cada valor do Dictionary são dois pontos ((x1, y1), (x2, y2)) que corresponde aos limites do grupo
    """

    width, height = np.shape(img)
    
    groups = np.zeros((width, height), dtype=np.uint8)
    bounds = {}
    current_group = 1

    queue = []

    for y in range(0, height):
        for x in range(0, width):
            if (img[x, y] == 0 and groups[x, y] == 0):
                groups[x, y] = current_group
                bounds[current_group] = ((y, x), (y, x))

                queue = queue + get_neighbours(img, (x, y), kernel)

                while queue:
                    pos_x, pos_y = queue.pop(0)

                    if (groups[pos_x, pos_y] == 0):
                        groups[pos_x, pos_y] = current_group

                        p_1 = bounds[current_group][0]
                        p_2 = bounds[current_group][1]

                        p_1 = (min(pos_y, p_1[0]), min(pos_x, p_1[1]))
                        p_2 = (max(pos_y, p_2[0]), max(pos_x, p_2[1]))

                        bounds[current_group] = (p_1, p_2)

                        queue = queue + get_neighbours(img, (pos_x, pos_y), kernel)

                current_group = current_group + 1

    return bounds

def get_neighbours(img, pos, kernel):
    """
    Obtem a vizinhança de um pixel baseado em uma Matriz de Congruência

    Parâmetros:
    - img: Matriz de pixels
    - pos: Posicao do pixel na imagem
    - kernel: Matriz de Congruência

    Retorno:
    Lista de tuplas (x, y) contendo a posição de cada vizinho do pixel de entrada
    """

    img_x, img_y = pos
    img_width, img_height = np.shape(img)
    kernel_width, kernel_height = np.shape(kernel)

    offset_x = math.floor(kernel_width / 2)
    offset_y = math.floor(kernel_height / 2)

    neighbours = []

    for kernel_y in range(0, kernel_height):
        for kernel_x in range(0, kernel_width):

            x = img_x + kernel_x - offset_x
            y = img_y + kernel_y - offset_y

            if (x < img_width and x >= 0 and y < img_height and y >= 0):
                if (kernel[kernel_x, kernel_y] == 1 and img[x, y] == 0):
                    neighbours.append((x, y))
    
    return neighbours      

def draw_bounds(img, bounds, color):
    """
    Desenha um retângulo colorido para cada grupo.

    Parâmetros:
    - img: Matriz de pixels
    - bounds: Dictionary de grupos onde cada valor corresponde à dois pontos. Ex.: ((x1, y1), (x2, y2))
    - color: Cor do retângulo. Ex.: (255, 0, 0)

    Retorno:
    Matriz de pixels.
    """
    
    for group in bounds.keys():
        p1 = bounds[group][0]
        p2 = bounds[group][1]

        cv.rectangle(img, p1, p2, color, 1)

    return img

def display_img(img):
    """
    Exibe uma imagem em uma nova janela.

    Parâmetros:
    - img: Matriz de pixels
    """

    plt.imshow(img)
    plt.show()
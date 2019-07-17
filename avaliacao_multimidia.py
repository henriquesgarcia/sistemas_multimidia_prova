'''
****** Avaliação de Sistemas Multimídia ******
********** Henrique Garcia - ADS V ***********
'''

import cv2, numpy as np

#-------------------------------------------------------------------
# Convertendo a imagem para tons de cinza

img_dados = cv2.imread('dados.jpg')

img_dados = cv2.cvtColor(img_dados, cv2.COLOR_BGR2GRAY)
cv2.imshow('Imagem em Tons de Cinza', img_dados)

# Salvando a imagem em disco:
cv2.imwrite('escala_de_cinza.jpg',img_dados)

cv2.waitKey(0)

#-------------------------------------------------------------------
# Aplicando filtro blur

img_cinza = cv2.imread('escala_de_cinza.jpg')
img_cinza = img_cinza[::1,::1] # Diminui a imagem

img_blur = np.vstack([
    np.hstack([cv2.blur(img_cinza, (5,5))]),
])

cv2.imshow("Imagem Suavizada (Blur)", img_blur)

# Salvando a imagem em disco:
cv2.imwrite('filtro_blur.jpg', img_blur)

cv2.waitKey(0)

#-------------------------------------------------------------------
# Aplicando binarização

img_filtro = cv2.imread('filtro_blur.jpg')

(T, bin) = cv2.threshold(img_filtro, 160, 255, cv2.THRESH_BINARY)

img_binarizada = np.vstack([
    np.hstack([bin]),
])

cv2.imshow("Binarizacao da Imagem", img_binarizada)

# Salvando a imagem em disco:
cv2.imwrite('binarizacao.jpg', img_binarizada)

cv2.waitKey(0)

#-------------------------------------------------------------------
# Detectando bordas

img_bin = cv2.imread('binarizacao.jpg')

img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)

img_bordas_detectadas = cv2.Canny(img_bin, 70, 200)

bordas_detectadas = np.vstack([
    np.hstack([img_bordas_detectadas]),
])

cv2.imshow("Deteccao de Bordas", bordas_detectadas)

# Salvando a imagem em disco:
cv2.imwrite('bordas_detectadas.jpg', bordas_detectadas)

cv2.waitKey(0)

# -------------------------------------------------------------------
# Contagem dos contornos externos

objetos, lx = cv2.findContours(img_bordas_detectadas,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

print('\nQuantidade de objetos na imagem: ' + str(len(objetos)))
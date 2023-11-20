#Lectura de Archivos 

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from funcionesPatentes import *

# Obtener la lista de archivos de imágenes en el directorio actual
archivos_imagen = [f for f in os.listdir('./patentes/') if f.endswith(('.png'))]
lista_imagenes= []
casos_especiales = []
for imagen in archivos_imagen:
  if imagen == "img01.png" or imagen =="img08.png" or imagen =="img11.png":
    casos_especiales.append(imagen)
  else: lista_imagenes.append(imagen)   

lista = procesamiento(lista_imagenes)
lista2= procesamiento_especial(casos_especiales)

imshow(lista2[2])

deteccionPatentes(lista)
deteccionPatentesEspeciales(lista2)
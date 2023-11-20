import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#################################################
#              FUNCIONES                    #####
#################################################

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)

def mostrar_imagenes(imagenes, titulos):
    num_imagenes = len(imagenes)
    num_filas = (num_imagenes + 1) // 2  # Si hay 1 o 2 imágenes, usa una fila; si hay 3 o 4, usa dos filas

    fig, axs = plt.subplots(num_filas, 2, figsize=(10, 8))

    for i, (img, title) in enumerate(zip(imagenes, titulos)):
        ax = axs[i // 2, i % 2] if num_imagenes > 1 else axs[i]
        ax.imshow(img, cmap='gray')  # Ajusta el cmap según el tipo de imagen (escala de grises o color)
        ax.set_title(title)
        ax.axis('off')

    plt.show()

        
def distancia(componente1, componente2):
  import math
  componente1= (componente1[0]+componente1[2], componente1[1])
  componente2= (componente2[0], componente2[1])
  distancia_e = math.sqrt((componente2[0] - componente1[0])**2 + (componente2[1] - componente1[1])**2)
  return distancia_e


def procesamiento (listaImagenes:list):
  lista_imagenes_procesadas = []

  #   lista_imagenes_procesadas.append(thresh)
  lista_imagenes_aumentadas=[]
  for nombre_archivo in listaImagenes:
      
      # Lee la imagen en formato BGR
      
      
      
      img_bgr = cv2.imread(f'./patentes/{nombre_archivo}')
      
      #img_bgr = cv2.imread(nombre_archivo)
      # Convierte la imagen a escala de grises
      img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


          # Ajusta el contraste utilizando la función addWeighted
      alpha = 1.5  # Factor de aumento de contraste
      beta = 0     # Desplazamiento

      contraste_aumentado = cv2.addWeighted(img_gray, alpha, np.zeros_like(img_gray), 0, beta)

          # Obtiene las dimensiones originales de la imagen
      alto, ancho = contraste_aumentado.shape[:2]

      # Define el nuevo tamaño (el doble del tamaño original)
      nuevo_ancho = ancho * 2
      nuevo_alto = alto * 2

      # Realiza el redimensionamiento utilizando cv2.resize
      img_redimensionada = cv2.resize(contraste_aumentado, (nuevo_ancho, nuevo_alto))

      #Guardar imagenes redimensionadas
      img_redimension = cv2.resize(img_gray, (nuevo_ancho, nuevo_alto))
      lista_imagenes_aumentadas.append(img_redimension)

      _, thresh = cv2.threshold(img_redimensionada, 0, 1,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
      #4 de 5, 5 de 6 con 119,190

      kernel = np.ones((15, 15), np.uint8)
      tophat = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel)

      kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
      # Kernel para erosión
      kernel2 = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]], dtype=np.uint8)

      # Aplicar erosión

      #lista_imagenes_procesadas.append(thresh)
      #imshow(tophat, title=nombre_archivo)

      erode = cv2.erode(tophat, kernel4,iterations=1)
      lista_imagenes_procesadas.append(erode)
      #imshow(tophat, title=nombre_archivo)
      mostrar_cuatro_imagenes([img_gray, thresh, tophat, erode],['Escala de Grises', 'Threshhold', 'TopHat', 'Erosionada'] )
  return lista_imagenes_procesadas


def deteccionPatentes(lista_imagenes_procesadas:list):

  indice = 0
  # deteccion de componentes

  for imagen in lista_imagenes_procesadas:

      imgContour = imagen

          # Encuentra componentes conectados
      num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imgContour, 4)

          # Especifica el umbral de área
      area_threshold = (50, 500)  # UMBRAL DE AREA

  # Detectar componentes conectados con stats en la imagen erosionada

            # Filtra las componentes conectadas basadas en el umbral de área
      filtered_labels = []
      filtered_stats = []
      filtered_centroids = []
      componentes_filtradas = np.zeros_like(imgContour)
      for label in range(1, num_labels):  # comienza desde 1 para excluir el fondo (etiqueta 0)
          area = stats[label, cv2.CC_STAT_AREA]

          if area > area_threshold[0] and area < area_threshold[1]:
              filtered_labels.append(label)
              filtered_stats.append(stats[label])
              filtered_centroids.append(centroids[label])
              componentes_filtradas[labels == label] = 255

    #   plt.imshow(imgContour, cmap='gray')

    #         # Dibuja los bounding boxes de las componentes conectadas filtradas
    #   for label, stat in zip(filtered_labels, filtered_stats):
    #         x, y, w, h, area = stat

    #         rect = plt.Rectangle((x, y), w, h, edgecolor='red', linewidth=2, fill=False)
    #         plt.gca().add_patch(rect)

      #imshow(componentes_filtradas, title='BB > area filtro')

      min_aspect_ratio = 0.40
      max_aspect_ratio = 0.70
      componentes_filtradas2 = np.zeros_like(imgContour)
        # Filtrar componentes conectadas por relación de aspecto
      filtered_stats_aspect = [stat for stat in filtered_stats if min_aspect_ratio < stat[2] / stat[3] < max_aspect_ratio]
      #for stat in filtered_stats_aspect:
        #print( stat[2] / stat[3])
      # Filtrar centroides por relación de aspecto usando las etiquetas ya filtradas
      filtered_centroids_aspect = [centroids[label] for label, stat in zip(filtered_labels, filtered_stats) if any(np.array_equal(stat, s) for s in filtered_stats_aspect)]
      #filtered_labels_aspect = [label for label, stat in zip(filtered_labels, filtered_stats) if any(np.array_equal(stat, s) for s in filtered_stats_aspect)]
      # Inicializa una lista vacía para almacenar las etiquetas filtradas
      filtered_labels_aspect = []
      # Itera sobre las etiquetas y estadísticas filtradas
      for label, stat in zip(filtered_labels, filtered_stats):
          # Verifica si alguna estadística en filtered_stats_aspect es igual a la estadística actual
          for s in filtered_stats_aspect:
              if np.array_equal(stat, s):
                  # Si es verdadero, agrega la etiqueta a la lista
                  filtered_labels_aspect.append(label)
                  componentes_filtradas2[labels == label] = 255
                  break  # Sale del bucle interno una vez que se encuentra una coincidencia}
      componentes_filtradas3=np.zeros_like(imgContour)
      umbral_distancia=30

      label_filtrada_distancia=[]

      for label, stat in zip(filtered_labels_aspect, filtered_stats_aspect):
        for s_label, s_stat in zip(filtered_labels_aspect, filtered_stats_aspect):
            if label != s_label and distancia(stat, s_stat) < umbral_distancia:
                #print(distancia(stat, s_stat))
                componentes_filtradas3[labels == label] = 255
                componentes_filtradas3[labels == s_label] = 255
                label_filtrada_distancia.append(label)
                break


      #imshow(componentes_filtradas3, title='BB > distancia')

      distancia_maxima = 200

      num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(componentes_filtradas3, 4,)
      # Calcular distancias entre centroides
      num_componentes = np.max(labels)
      distancias = np.zeros((num_componentes + 1, num_componentes + 1), dtype=np.float32)

      for i in range(1, num_componentes + 1):
          for j in range(i + 1, num_componentes + 1):
              distancias[i, j] = np.linalg.norm(centroids[i] - centroids[j])
              distancias[j, i] = distancias[i, j]

      # Filtrar componentes conectados
      componentes_filtradas = []

      for i in range(1, num_componentes + 1):
          componentes_adyacentes = np.where(distancias[i] < distancia_maxima)[0]

          # Incluye el componente actual solo si tiene al menos 5 componentes adyacentes
          if len(componentes_adyacentes) >= 5:
              componentes_filtradas.append(i)

      # Crear una imagen para visualizar las componentes conectadas filtradas
      imagen_resultado = np.zeros_like(labels, dtype=np.uint8)

      for componente in componentes_filtradas:
          # Selecciona solo las píxeles de la componente actual
          mascara_componente = np.uint8(labels == componente)
          imagen_resultado += mascara_componente * componente


      # Visualizar la imagen resultante
      #imshow(imagen_resultado,title='Componentes Conectados Filtrados')

      num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_resultado, 4,)

      x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(np.uint8(labels > 0))

      # Recortar el rectángulo de la imagen original
      imagen_recortada = lista_imagenes_procesadas[indice][y_rect - 10 : y_rect + h_rect + 10, x_rect - 10 : x_rect + w_rect + 10]
      
      mostrar_cuatro_imagenes([imgContour, componentes_filtradas3, imagen_resultado, imagen_recortada],['Procesada', 'Filtro area + aspect', 'Caracteres patente', 'Patente'] )

      #imshow(imagen_recortada ,  title="patente segmentada")

      indice +=1
      


#### Casos especiales

def procesamiento_especial(archivos_imagen:list):
  lista_imagenes_procesadas = []
  # Iterar sobre cada imagen


  for nombre_archivo in archivos_imagen:


      # Lee la imagen en formato BGR
      #img_bgr = cv2.imread(nombre_archivo)
      img_bgr = cv2.imread(f'./patentes/{nombre_archivo}')
      # Convierte la imagen a escala de grises
      img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


          # Ajusta el contraste utilizando la función addWeighted
      alpha = 1.5  # Factor de aumento de contraste
      beta = 0     # Desplazamiento

      contraste_aumentado = cv2.addWeighted(img_gray, alpha, np.zeros_like(img_gray), 0, beta)

      if nombre_archivo == 'img11.png':

          _, thresh = cv2.threshold(contraste_aumentado, 205, 255, cv2.THRESH_BINARY)
          lista_imagenes_procesadas.append(thresh)
          mostrar_cuatro_imagenes([img_gray, thresh],['Escala de Grises', 'Threshhold'] )
          continue
          # Obtiene las dimensiones originales de la imagen
      alto, ancho = contraste_aumentado.shape[:2]

      # Define el nuevo tamaño (el doble del tamaño original)
      nuevo_ancho = ancho * 3
      nuevo_alto = alto * 3

      # Realiza el redimensionamiento utilizando cv2.resize
      img_redimensionada = cv2.resize(contraste_aumentado, (nuevo_ancho, nuevo_alto))

      _, thresh = cv2.threshold(img_redimensionada, 0, 1,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

      
      kernel = np.ones((15, 15), np.uint8)
      if nombre_archivo == 'img01.png':
        tophat = cv2.morphologyEx(thresh, cv2.MORPH_TOPHAT, kernel)#funciona para imagen 01 + erode!

      kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))#funciona para imagen 08


      # Aplicar erosión
      img_eroded = cv2.erode(thresh, kernel2, iterations=3)

      lista_imagenes_procesadas.append(img_eroded)
      #imshow(tophat, title=nombre_archivo)
      mostrar_cuatro_imagenes([img_gray, thresh, tophat, img_eroded],['Escala de Grises', 'Threshhold', 'TopHat', 'Erosionada'] )

  
  return lista_imagenes_procesadas


def deteccionPatentesEspeciales(lista_imagenes_procesadas:list):

  indice = 0
  # deteccion de componentes

  for imagen in lista_imagenes_procesadas:

      imgContour = imagen
      #imshow(imgContour, title=f'{indice}')
          # Encuentra componentes conectados
      num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imgContour, 4)

          # Especifica el umbral de área
      #area_threshold = (50, 750)  # UMBRAL DE AREA
      #area_threshold2 = (25,500)
  # Detectar componentes conectados con stats en la imagen erosionada

            # Filtra las componentes conectadas basadas en el umbral de área
      filtered_labels = []
      filtered_stats = []
      filtered_centroids = []
      componentes_filtradas = np.zeros_like(imgContour)


      if indice == 2:
        area_threshold = (20,100)  # UMBRAL DE AREA
      else:area_threshold = (50, 750)  # UMBRAL DE AREA


      for label in range(1, num_labels):  # comienza desde 1 para excluir el fondo (etiqueta 0)
          area = stats[label, cv2.CC_STAT_AREA]

          if area > area_threshold[0] and area < area_threshold[1]:
              filtered_labels.append(label)
              filtered_stats.append(stats[label])
              filtered_centroids.append(centroids[label])
              componentes_filtradas[labels == label] = 255

     # plt.imshow(imgContour, cmap='gray')

    #         # Dibuja los bounding boxes de las componentes conectadas filtradas
    #   for label, stat in zip(filtered_labels, filtered_stats):
    #         x, y, w, h, area = stat

    #         rect = plt.Rectangle((x, y), w, h, edgecolor='red', linewidth=2, fill=False)
    #         plt.gca().add_patch(rect)

      #imshow(componentes_filtradas, title='BB > area filtro')

      min_aspect_ratio = 0.30
      max_aspect_ratio = 0.70
      componentes_filtradas2 = np.zeros_like(imgContour)
        # Filtrar componentes conectadas por relación de aspecto
      filtered_stats_aspect = [stat for stat in filtered_stats if min_aspect_ratio < stat[2] / stat[3] < max_aspect_ratio]
      #for stat in filtered_stats_aspect:
        #print( stat[2] / stat[3])
      # Filtrar centroides por relación de aspecto usando las etiquetas ya filtradas
      filtered_centroids_aspect = [centroids[label] for label, stat in zip(filtered_labels, filtered_stats) if any(np.array_equal(stat, s) for s in filtered_stats_aspect)]
      #filtered_labels_aspect = [label for label, stat in zip(filtered_labels, filtered_stats) if any(np.array_equal(stat, s) for s in filtered_stats_aspect)]
      # Inicializa una lista vacía para almacenar las etiquetas filtradas
      filtered_labels_aspect = []
      # Itera sobre las etiquetas y estadísticas filtradas
      for label, stat in zip(filtered_labels, filtered_stats):
          # Verifica si alguna estadística en filtered_stats_aspect es igual a la estadística actual
          for s in filtered_stats_aspect:
              if np.array_equal(stat, s):
                  # Si es verdadero, agrega la etiqueta a la lista
                  filtered_labels_aspect.append(label)
                  componentes_filtradas2[labels == label] = 255
                  break  # Sale del bucle interno una vez que se encuentra una coincidencia}
      componentes_filtradas3=np.zeros_like(imgContour)
      umbral_distancia=30

      label_filtrada_distancia=[]

      for label, stat in zip(filtered_labels_aspect, filtered_stats_aspect):
        for s_label, s_stat in zip(filtered_labels_aspect, filtered_stats_aspect):
            if label != s_label and distancia(stat, s_stat) < umbral_distancia:
                #print(distancia(stat, s_stat))
                componentes_filtradas3[labels == label] = 255
                componentes_filtradas3[labels == s_label] = 255
                label_filtrada_distancia.append(label)
                break


      #imshow(componentes_filtradas3, title='BB > distancia')

      distancia_maxima = 200

      num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(componentes_filtradas3, 4,)
      # Calcular distancias entre centroides
      num_componentes = np.max(labels)
      distancias = np.zeros((num_componentes + 1, num_componentes + 1), dtype=np.float32)

      for i in range(1, num_componentes + 1):
          for j in range(i + 1, num_componentes + 1):
              distancias[i, j] = np.linalg.norm(centroids[i] - centroids[j])
              distancias[j, i] = distancias[i, j]

      # Filtrar componentes conectados
      componentes_filtradas = []

      for i in range(1, num_componentes + 1):
          componentes_adyacentes = np.where(distancias[i] < distancia_maxima)[0]

          # Incluye el componente actual solo si tiene al menos 5 componentes adyacentes
          if len(componentes_adyacentes) >= 5:
              componentes_filtradas.append(i)

      # Crear una imagen para visualizar las componentes conectadas filtradas
      imagen_resultado = np.zeros_like(labels, dtype=np.uint8)

      for componente in componentes_filtradas:
          # Selecciona solo las píxeles de la componente actual
          mascara_componente = np.uint8(labels == componente)
          imagen_resultado += mascara_componente * componente


      # Visualizar la imagen resultante
      #imshow(imagen_resultado,title='Componentes Conectados Filtrados')

      num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_resultado, 4,)

      x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(np.uint8(labels > 0))

      # Recortar el rectángulo de la imagen original
      imagen_recortada = lista_imagenes_procesadas[indice][y_rect - 20 : y_rect + h_rect + 20, x_rect - 20 : x_rect + w_rect + 20]
      #imshow(imagen_recortada)

      #imshow(imagen_recortada ,  title="patente segmentada")

      indice +=1
      mostrar_cuatro_imagenes([imgContour, componentes_filtradas3, imagen_resultado, imagen_recortada],['Procesada', 'Filtro area + aspect', 'Caracteres patente', 'Patente'] )

#################################################
#                                           #####
#################################################


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



deteccionPatentes(lista)
deteccionPatentesEspeciales(lista2)


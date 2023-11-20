import cv2
import numpy as np
import matplotlib.pyplot as plt


# Declaracion de funciones

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


def imreconstruct(marker, mask, kernel=None):
    if kernel==None:
        kernel = np.ones((3,3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               # Dilatacion
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   # Interseccion
        if (marker == expanded_intersection).all():                         # Finalizacion?
            break                                                           #
        marker = expanded_intersection
    return expanded_intersection

def imfillhole(img):
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    mask = np.zeros_like(img)                                                   # Genero mascara para...
    mask = cv2.copyMakeBorder(mask[1:-1,1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)) # ... seleccionar los bordes.
    marker = cv2.bitwise_not(img, mask=mask)                # El marcador lo defino como el complemento de los bordes.
    img_c = cv2.bitwise_not(img)                            # La mascara la defino como el complemento de la imagen.
    img_r = imreconstruct(marker=marker, mask=img_c)        # Calculo la reconstrucción R_{f^c}(f_m)
    img_fh = cv2.bitwise_not(img_r)                         # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.
    return img_fh


def contar_dados(contorno, imagen) -> int:

    mask = np.zeros_like(imagen)

    # Dibuja el contorno en la máscara
    cv2.drawContours(mask, [contorno], -1, 255, thickness=cv2.FILLED)

    # Aplica la máscara a la imagen original
    dado_recortado = cv2.bitwise_and(imagen, imagen, mask=mask)

    dado_recortado_u =  cv2.threshold(dado_recortado, 168, 255, cv2.THRESH_BINARY_INV)[1]
    dado_recortado_c = cv2.Canny(dado_recortado_u, 0, 255, apertureSize=3, L2gradient=True)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))

    dado_recortado_para_components = cv2.dilate(dado_recortado_c, kernel)

    # Encuentra componentes conectados
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dado_recortado_para_components, 8)

    # Especifica el umbral de área
    area_threshold = (700, 3000)  # UMBRAL DE AREA

    # Filtra las componentes conectadas basadas en el umbral de área
    filtered_labels = []
    filtered_stats = []
    filtered_centroids = []

    for label in range(1, num_labels):  # comienza desde 1 para excluir el fondo (etiqueta 0)
        area = stats[label, cv2.CC_STAT_AREA]

        if area > area_threshold[0] and area < area_threshold[1]:
            filtered_labels.append(label)
            filtered_stats.append(stats[label])
            filtered_centroids.append(centroids[label])
    return len(filtered_centroids)


def contar_moneda(area):

    if area > 69000 and area < 80000:
        return 10
    elif area > 80000 and area < 110000:
        return 1
    else:
        return 50



###################################################################
# Programa principal


img = cv2.imread('./monedas.jpg', cv2.IMREAD_GRAYSCALE)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#-------------------- BLUR PARA HOMOGENEIZAR FONDO ------------------

img_blur = cv2.medianBlur(img, 9, 2)
# plt.figure()
# ax1 = plt.subplot(121); plt.imshow(img, cmap="gray"), plt.title("Imagen")
# plt.subplot(122, sharex=ax1, sharey=ax1), plt.imshow(img_blur, cmap="gray"), plt.title("Imagen + blur")
# plt.show(block=False)

# --------------------------------- CANNY --------------------------------

# img_canny_CV2 = cv2.Canny(img_blur, 50, 115, apertureSize=3, L2gradient=True)
img_canny_CV2 = cv2.Canny(img_blur, 10, 54, apertureSize=3, L2gradient=True)
#imshow(img_canny_CV2)


f=img_canny_CV2.copy()


# --------------------------- DILATACION Y CLAUSURA -------------------

k = 22
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))

fd = cv2.dilate(f, kernel)
fc = cv2.morphologyEx(fd, cv2.MORPH_CLOSE, (7,7))

# fe = cv2.erode(fd, 3)
#imshow(fc, title= 'Dilatacion + Clausura')


#----------------------RELLENADO DE HUECOS FUNCION + APERTURA---------------------------------

rellenada=imfillhole(fc)
#imshow(rellenada, title='Rellenada')

rellenada2=rellenada.copy()

kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(121,121))

rConClausura = cv2.morphologyEx(rellenada2, cv2.MORPH_OPEN, kernel3)

#imshow(rConClausura,title='Rellenada + Clausura')

#-------------------------CONTORNOS PARA SEPARAR Y CLASIFICAR ELEMENTOS------------------------


imgContour = rConClausura.copy()

# Encuentra componentes conectados
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imgContour, 8)

# Especifica el umbral de área
area_threshold = 300  # UMBRAL DE AREA

# Filtra las componentes conectadas basadas en el umbral de área
filtered_labels = []
filtered_stats = []
filtered_centroids = []

for label in range(1, num_labels):  # comienza desde 1 para excluir el fondo (etiqueta 0)
    area = stats[label, cv2.CC_STAT_AREA]

    if area > area_threshold:
        filtered_labels.append(label)
        filtered_stats.append(stats[label])
        filtered_centroids.append(centroids[label])

# Convierte las listas filtradas a matrices
filtered_labels = np.array(filtered_labels)
filtered_stats = np.array(filtered_stats)
filtered_centroids = np.array(filtered_centroids)



####################################################CLASIFICACION##########################################



cnt, _ = cv2.findContours(rConClausura, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
objetos = (labels).astype(np.uint8)
cnt, _ = cv2.findContours(objetos, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


labeled_shapes = np.zeros_like(rConClausura)

dados = []
monedas = []
factdeforma = []

for i in range(1, num_labels):
    objeto = (labels == i).astype(np.uint8)
    cont, _ = cv2.findContours(objeto, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    if cont:
        for c in cont:
            dados_d={}
            monedas_d={}
            area = cv2.contourArea(c)
            p = cv2.arcLength(c, True)
            fp = 1 / (area / p ** 2)
            #print(i, cont)
            factdeforma.append(fp)

            if 14 <= fp < 15:
                monedas_d['area']=area
                monedas_d['img']=objeto
                monedas_d['contorno']=c
                monedas_d['valor']=contar_moneda(area)
                monedas.append(monedas_d)

            else:
                dados_d['fp']=fp
                dados_d['img']=objeto
                dados_d['contorno']=c
                dados_d['valor']=contar_dados(c, img)
                dados.append(dados_d)

#DEBUG

# for x in range(len(monedas)):
#     print(monedas[x]['valor'])


# Dibuja los bounding boxes en la imagen

for i in range(19):
    x, y, w, h, area = filtered_stats[i]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Muestra el resultado
# plt.imshow(img)
# plt.title('Original + BB')
# plt.show()


###################################################################
# Resultado Final

img = cv2.imread('./monedas.jpg',cv2.IMREAD_COLOR)

# Crear una figura con 2 filas y 2 columnas para mostrar las imágenes
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Subfigura 1: Monedas con valor 50
axs[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0, 0].set_title('Monedas con Valor .50')
for moneda in monedas:
    if moneda['valor'] == 50:
        x, y, w, h = cv2.boundingRect(moneda['contorno'])
        axs[0, 0].add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='g', facecolor='none'))
        axs[0, 0].text(x, y, f"50", color='g', fontsize=10)
axs[0, 0].text(2500,2500,f"Cantidad de monedas: {len([m for m in monedas if m['valor'] == 50])}", color='g', fontsize=10, ha='center')

# Subfigura 2: Monedas con valor 1
axs[0, 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0, 1].set_title('Monedas con Valor 1')
for moneda in monedas:
    if moneda['valor'] == 1:
        x, y, w, h = cv2.boundingRect(moneda['contorno'])
        axs[0, 1].add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none'))
        axs[0, 1].text(x, y, f"1", color='r', fontsize=10)
axs[0, 1].text(2500,2500,f"Cantidad de monedas: {len([m for m in monedas if m['valor'] == 1])}", color='r', fontsize=10, ha='center')

# Subfigura 3: Monedas con valor 10
axs[1, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[1, 0].set_title('Monedas con Valor 10')
for moneda in monedas:
    if moneda['valor'] == 10:
        x, y, w, h = cv2.boundingRect(moneda['contorno'])
        axs[1, 0].add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='b', facecolor='none'))
        axs[1, 0].text(x, y, f"10", color='b', fontsize=10)
axs[1, 0].text(2500,2500,f"Cantidad de monedas: {len([m for m in monedas if m['valor'] == 10])}", color='b', fontsize=10, ha='center')
# Subfigura 4: Dados
axs[1, 1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[1, 1].set_title('Dados')
for dado in dados:
    x, y, w, h = cv2.boundingRect(dado['contorno'])
    axs[1, 1].add_patch(plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='m', facecolor='none'))
    axs[1, 1].text(x, y, f"{dado['valor']}", color='m', fontsize=10)

plt.show()
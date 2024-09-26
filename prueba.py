import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

# Construir la red neuronal
# La red se representará mediante una matriz de adyacencia, donde cada fila guarda los vecinos del nodo actual
# Las conexiones entre nodos serán aleatorias
def create_network(num_nodes=100, connection_probability=0.5):
    network = []  # Lista para almacenar los vecinos de cada nodo
    weights_matrix = []  # Matriz para almacenar los pesos de las conexiones
    
    for _ in range(num_nodes):
        # Generar conexiones aleatorias usando una distribución de Bernoulli
        neighbors = bernoulli.rvs(connection_probability, size=num_nodes)
        neighbors = np.where(neighbors == 1)[0]  # Obtener los índices de los nodos vecinos
        
        # Almacenar las conexiones y sus pesos (inicializados en cero)
        network.append(neighbors)
        weights_matrix.append(np.zeros(len(neighbors)))
                
    return network, weights_matrix

# Aprender un patrón dado una red y un patrón específico
def learn(network, weights_matrix, pattern):
    # Recorrer los vecinos de cada nodo para actualizar los pesos
    for i, neighbors in enumerate(network):
        # Regla de aprendizaje de Hebb
        weights_matrix[i] += pattern[neighbors] * pattern[i]

# Convertir un índice lineal a coordenadas (x, y) en una matriz dada su ancho
def vect2matrix(index, width):
    y = index // width
    x = index - y * width
    return x, y

##############
# EJECUCIÓN PRINCIPAL
##############

#########################
# ABRIR LA IMAGEN
###########################

# Cargar una imagen en escala de grises
img = cv2.imread("kanekaBW.png", 0)
img = cv2.resize(img, (100, 100))  # Redimensionar la imagen a 100x100 píxeles

# Calcular el umbral para binarizar la imagen
threshold = img.mean()
img[np.where(img < threshold)] = 0  # Asignar 0 a los píxeles por debajo del umbral
img[np.where(img >= threshold)] = 1  # Asignar 1 a los píxeles por encima del umbral

# Convertir la imagen a un arreglo de tipo int8
img = np.array(img, dtype='int8')
img[np.where(img == 0)] = -1  # Convertir los valores 0 a -1 para el procesamiento

# Aplanar la imagen para la red neuronal
flat_img = np.resize(img, img.size)

############################
# CREACIÓN DE LA RED
############################

# Crear la red neuronal con el número de nodos igual al tamaño de la imagen
network, weights_matrix = create_network(num_nodes=len(flat_img), connection_probability=0.3)

# Aprender el patrón de la imagen
learn(network, weights_matrix, flat_img)

#############################
# RECONSTRUCCIÓN DE LA IMAGEN
#############################

# Generar una imagen aleatoria con ruido
random_img = flat_img.copy()
noise = bernoulli.rvs(0.8, size=len(random_img))    
random_img[np.where(noise == 1)] = -1  # Cambiar algunos píxeles a -1
noise = bernoulli.rvs(0.5, size=len(random_img))    
random_img[np.where(noise == 1)] = 1  # Cambiar algunos píxeles a 1
noise_img = np.resize(random_img, img.shape)

# Ajustar los valores de la imagen con ruido para visualización
noise_img[np.where(noise_img == 1)] = 120
noise_img[np.where(noise_img == -1)] = 0
width = len(noise_img)

# Las neuronas utilizarán los pesos para reconstruir la imagen
for i, neighbors in enumerate(network):
    # Calcular el nuevo estado basándose en el estado de los vecinos y sus pesos
    new_state = sum(random_img[neighbors] * weights_matrix[i])
    
    # Definir el nuevo estado en función de la suma de los vecinos
    if new_state < 0:
        new_state = 0
    else:
        new_state = 120

    # Convertir el índice del nodo a coordenadas (x, y)
    x, y = vect2matrix(i, width)

    # Actualizar la matriz de imagen con el nuevo estado
    noise_img[y, x] = new_state

    # Mostrar la imagen reconstruida en una ventana redimensionable
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Frame', 1000, 1000)  
    cv2.imshow('Frame', noise_img) 

    # Ciclo principal de visualización
    keyboard = cv2.waitKey(30)

    # Salir del ciclo si se presiona 'q' o 'Esc'
    if keyboard == ord('q') or keyboard == 27:
        break

print("Presiona cualquier tecla para cerrar la ventana...")

# Mantener la ventana abierta hasta que se presione una tecla
cv2.waitKey(0)

# Cerrar todas las ventanas de OpenCV
cv2.destroyAllWindows()

import numpy as np
import random
import cv2

# Cargar la imagen deseada desde un archivo local
imagen_deseada = cv2.imread('kanekaBW.png')  # Usa tu imagen local
imagen_deseada = cv2.resize(imagen_deseada, (250, 250))  # Redimensionar a 250x250

# Convertir la imagen deseada a escala de grises
imagen_deseada_gris = cv2.cvtColor(imagen_deseada, cv2.COLOR_BGR2GRAY)

# Crear una imagen inicial blanca de 250x250
imagen_inicial = np.ones((250, 250), dtype=np.uint8) * 255

# Función de adaptación
def calcular_adaptacion(imagen_array):
    diff_array = np.abs(imagen_deseada_gris - imagen_array)
    adaptacion = np.sum(diff_array)
    return adaptacion

# Función de mutación
def funcion_mutacion(imagen_array):
    mutacion_array = np.copy(imagen_array)
    for i in range(10):  # Realizar 10 mutaciones aleatorias
        x = random.randint(0, imagen_array.shape[0] - 1)
        y = random.randint(0, imagen_array.shape[1] - 1)
        mutacion_array[x, y] = random.randint(0, 255)
    return mutacion_array

# Función de selección
def funcion_seleccion(poblacion, adaptacion, num_padres):
    ganadores = []
    for i in range(num_padres):
        torneo_indices = random.sample(range(len(poblacion)), 5)
        torneo_adaptacion = [adaptacion[j] for j in torneo_indices]
        ganador = torneo_indices[torneo_adaptacion.index(min(torneo_adaptacion))]
        ganadores.append(ganador)
    return [poblacion[i] for i in ganadores]

# Función principal del algoritmo genético
def ejecutar_algoritmo():
    tam_poblacion = 1000  # Tamaño de la población
    num_padres = 20       # Número de padres seleccionados
    num_iteraciones = 20000  # Número de iteraciones

    poblacion_actual = [imagen_inicial] * tam_poblacion
    for i in range(num_iteraciones):
        puntos = [calcular_adaptacion(image) for image in poblacion_actual]
        padres = funcion_seleccion(poblacion_actual, puntos, num_padres)
        hijos = []

        for padre in padres:
            mutacion = funcion_mutacion(padre)
            hijos.append(mutacion)

        poblacion_actual = padres + hijos

        # Mostrar la mejor imagen cada 100 iteraciones
        if i % 100 == 0:
            mejor_indice = puntos.index(min(puntos))
            mejor_imagen = poblacion_actual[mejor_indice]
            cv2.imshow('Imagen Evolucionada', mejor_imagen)
            cv2.waitKey(1)  # Actualiza cada iteración

    # Mostrar la mejor imagen final
    mejor_indice = puntos.index(min(puntos))
    mejor_imagen = poblacion_actual[mejor_indice]
    cv2.imshow('Imagen Final Evolucionada', mejor_imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ejecutar el algoritmo genético
ejecutar_algoritmo()

import numpy as np
import time
import matplotlib.pyplot as plt

# Contadores globales
mult_count = 0
div_count = 0
add_sub_count = 0

def reset_counters():
    global mult_count, div_count, add_sub_count
    mult_count = 0
    div_count = 0
    add_sub_count = 0

def imprimir_resultados(nombre_metodo, solucion, tiempo):
    print(f"\n--- {nombre_metodo} ---")
    print("Solución:", solucion.flatten())
    print(f"Tiempo de ejecución: {tiempo:.6f} segundos")
    print(f"Multiplicaciones/Divisiones: {mult_count + div_count}")
    print(f"Sumas/Restas: {add_sub_count}")

def eliminacion_gaussiana(A: np.ndarray) -> np.ndarray:
    global mult_count, div_count, add_sub_count
    reset_counters()
    inicio = time.time()

    A = np.array(A, dtype=float)
    n = A.shape[0]

    for i in range(n - 1):
        p = None
        for pi in range(i, n):
            if A[pi, i] == 0:
                continue
            if p is None or abs(A[pi, i]) < abs(A[p, i]):
                p = pi
        if p is None:
            raise ValueError("No existe solución única.")
        if p != i:
            A[[i, p]] = A[[p, i]]

        for j in range(i + 1, n):
            m = A[j, i] / A[i, i]
            div_count += 1
            for k in range(i, n + 1):
                A[j, k] -= m * A[i, k]
                mult_count += 1
                add_sub_count += 1

    solucion = np.zeros(n)
    solucion[n - 1] = A[n - 1, n] / A[n - 1, n - 1]
    div_count += 1

    for i in range(n - 2, -1, -1):
        suma = 0
        for j in range(i + 1, n):
            suma += A[i, j] * solucion[j]
            mult_count += 1
            add_sub_count += 1
        solucion[i] = (A[i, n] - suma) / A[i, i]
        div_count += 1
        add_sub_count += 1

    fin = time.time()
    imprimir_resultados("Eliminación Gaussiana", solucion, fin - inicio)
    return solucion

def gauss_jordan(Ab: np.ndarray) -> np.ndarray:
    global mult_count, div_count, add_sub_count
    reset_counters()
    inicio = time.time()

    Ab = np.array(Ab, dtype=float)
    n = Ab.shape[0]

    for i in range(n):
        p = None
        for pi in range(i, n):
            if Ab[pi, i] == 0:
                continue
            if p is None or abs(Ab[pi, i]) < abs(Ab[p, i]):
                p = pi
        if p is None:
            raise ValueError("No existe solución única.")
        if p != i:
            Ab[[i, p]] = Ab[[p, i]]

        for j in range(n):
            if i == j:
                continue
            m = Ab[j, i] / Ab[i, i]
            div_count += 1
            for k in range(i, n + 1):
                Ab[j, k] -= m * Ab[i, k]
                mult_count += 1
                add_sub_count += 1

    solucion = np.zeros(n)
    for i in range(n):
        solucion[i] = Ab[i, n] / Ab[i, i]
        div_count += 1

    fin = time.time()
    imprimir_resultados("Gauss-Jordan", solucion, fin - inicio)
    return solucion

def comparar_metodos(Ab_original, metodo_gauss, metodo_jordan):
    # Hacer copia de la matriz original para no modificarla
    Ab1 = np.array(Ab_original, dtype=float)
    Ab2 = np.array(Ab_original, dtype=float)

    # Ejecutar Eliminación Gaussiana
    start = time.time()
    metodo_gauss(np.copy(Ab1))
    end = time.time()
    tiempo_gauss = end - start
    operaciones_gauss = metodo_gauss.mult_count + metodo_gauss.div_count + metodo_gauss.add_sub_count

    # Ejecutar Gauss-Jordan
    start = time.time()
    metodo_jordan(np.copy(Ab2))
    end = time.time()
    tiempo_jordan = end - start
    operaciones_jordan = metodo_jordan.mult_count + metodo_jordan.div_count + metodo_jordan.add_sub_count

    # Crear gráfico
    plt.figure(figsize=(8, 5))
    plt.scatter(operaciones_gauss, tiempo_gauss, color='blue', label='Eliminación Gaussiana')
    plt.scatter(operaciones_jordan, tiempo_jordan, color='red', label='Gauss-Jordan')
    plt.xlabel("Número total de operaciones")
    plt.ylabel("Tiempo de ejecución (segundos)")
    plt.title("Comparación: Tiempo vs Operaciones")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def get_counters():
    return mult_count, div_count, add_sub_count

# Para graficar
eliminacion_gaussiana.mult_count = lambda: mult_count
eliminacion_gaussiana.div_count = lambda: div_count
eliminacion_gaussiana.add_sub_count = lambda: add_sub_count

def get_total_operations():
    return mult_count + div_count + add_sub_count
def get_time_last():
    return tiempo_global if 'tiempo_global' in globals() else 0
import numpy as np
import matplotlib.pyplot as plt


def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    n = len(b)
    # Valor incial inicia en 0 por default
    x = np.zeros(n)
    x_prev = np.copy(x)
    errors = []
    
    for k in range(max_iter):
        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x_prev[j] for j in range(i + 1, n))
            x[i] = (b[i] - sum1 - sum2) / A[i][i]
        
        abs_error = np.linalg.norm(x - x_prev, ord=np.inf)
        rel_error = abs_error / (np.linalg.norm(x, ord=np.inf) + 1e-10)
        quad_error = np.linalg.norm(x - x_prev) ** 2
        
        errors.append((k, abs_error, rel_error, quad_error))
        print(f"Iteración {k+1}: Error absoluto = {abs_error}, Error relativo = {rel_error}, Error cuadrático = {quad_error}")

        
        if abs_error < tol:
            break
        
        x_prev = np.copy(x)
    
    return x, errors
    #10I1 +2I2 +3I3 +I4 = 15
 #2I1 +12I2 +2I3 +3I4 = 22
 #3I1 +2I2 +15I3 +I4 = 18
 #I1 +3I2 +I3 +10I4 = 10

A = np.array([
    [10, 2, 3, 1],
    [2, 12, 2, 3],
    [3, 2, 15, 1],
    [1, 3, 1, 10]
])

b = np.array([15, 22, 18, 10])

# Llama a funcion de Gauss-Seidel
x_sol, errors = gauss_seidel(A, b)
print (f"Soluciones aproximadas: {x_sol}")


# Graficar errores
iterations = [e[0] for e in errors]
abs_errors = [e[1] for e in errors]
rel_errors = [e[2] for e in errors]
quad_errors = [e[3] for e in errors]

plt.figure(figsize=(10, 5))
plt.plot(iterations, abs_errors, label="Error absoluto")
plt.plot(iterations, rel_errors, label="Error relativo")
plt.plot(iterations, quad_errors, label="Error cuadrático")
plt.yscale("log")
plt.xlabel("Iteraciones")
plt.ylabel("Errores")
plt.title("Convergencia del método de Gauss-Seidel")
plt.legend()
plt.grid()
plt.savefig("convergencia_gauss_seidel.png")
plt.show()
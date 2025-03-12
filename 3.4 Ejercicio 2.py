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
    
 #20T1 −5T2 −3T3 = 100
 #−4T1 +18T2 −2T3 −T4 = 120
 #−3T1 −T2 +22T3 −5T4 = 130
 #−2T2 −4T3 +25T4 −T5 = 150

A = np.array([
    [20, -5, -3, 0, 0],
    [-4, 18, -2, -1, 0],
    [-3, -1, 22, -5, 0],
    [0, -2, -4, 25, -1],
    [0,0,0,0,1]
])

b = np.array([100, 120, 130, 150,1])

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
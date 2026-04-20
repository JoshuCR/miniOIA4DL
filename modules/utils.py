# PISTA: es esta la mejor forma de hacer una matmul?
def matmul_biasses_naive(A, B, C, bias):
    m, p, n = A.shape[0], A.shape[1], B.shape[1]
    for i in range(m):
        for j in range(n):
            for k in range(p):
                C[i][j] += A[i][k] * B[k][j]
            C[i][j] += bias[j]
    return C

# --- INICIO BLOQUE GENERADO CON IA ---
# Versión 2 - Loop reordered: mismos 3 bucles pero en orden i,k,j
# Al poner k antes que j, accedemos a B[k][j] de forma continua en memoria.
# Referencia transparencias "GEMM - Loop reordered - 2.5ms"
def matmul_biasses_reordered(A, B, C, bias):
    m, p, n = A.shape[0], A.shape[1], B.shape[1]
    for i in range(m):
        for k in range(p):
            for j in range(n):
                C[i][j] += A[i][k] * B[k][j]
    C += bias
    return C

# Versión 3 - NumPy BLAS: una sola operación NumPy
# NumPy usa BLAS internamente
# Referencia transparencias "GEMM - NumPy - 1.5ms"
def matmul_biasses(A, B, C, bias):
    return A @ B + bias
# --- FIN BLOQUE GENERADO CON IA ---
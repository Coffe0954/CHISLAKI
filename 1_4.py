import math

def read_input_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except UnicodeDecodeError:
        with open(filename, 'r', encoding='cp1251') as f:
            lines = [line.strip() for line in f if line.strip()]
            
    n = int(lines[0])
    A = []
    for i in range(1, n + 1):
        row = list(map(float, lines[i].split()))
        A.append(row)
    return n, A

def print_matrix(matrix):
    for row in matrix:
        print("  ".join(f"{val:10.4f}" for val in row))
    print()

def transpose(matrix):
    return [list(row) for row in zip(*matrix)]

def multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    res = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            res[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    return res

def jacobi_method(A, eps=0.01):
    n = len(A)
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    curr_A = [row[:] for row in A]
    
    while True:
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(curr_A[i][j]) > max_val:
                    max_val = abs(curr_A[i][j])
                    p, q = i, j
                    
        if max_val < eps:
            break
            
        if abs(curr_A[p][p] - curr_A[q][q]) < 1e-12:
            phi = math.pi / 4.0
        else:
            phi = 0.5 * math.atan(2.0 * curr_A[p][q] / (curr_A[p][p] - curr_A[q][q]))
            
        c = math.cos(phi)
        s = math.sin(phi)
        
        U = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        U[p][p] = c
        U[q][q] = c
        U[p][q] = -s
        U[q][p] = s
        
        V = multiply(V, U)
        U_T = transpose(U)
        curr_A = multiply(multiply(U_T, curr_A), U)
        
    eigenvalues = [curr_A[i][i] for i in range(n)]
    return eigenvalues, V

if __name__ == "__main__":
    filename = "input_lr1_4.txt"
    n, A = read_input_file(filename)

    print("=== LAB 1.4 ===")
    print("Initial Matrix A:")
    print_matrix(A)

    eigenvalues, V = jacobi_method(A, 0.01)

    print("--- Eigenvalues (Lambda) ---")
    for idx, val in enumerate(eigenvalues, 1):
        print(f"lambda_{idx} = {val:.6f}")
    print()

    print("--- Eigenvectors Matrix V ---")
    print_matrix(V)

    Lambda = [[0.0]*n for _ in range(n)]
    for i in range(n):
        Lambda[i][i] = eigenvalues[i]

    AV = multiply(A, V)
    VLambda = multiply(V, Lambda)

    print("--- Check: A * V ---")
    print_matrix(AV)
    print("--- Check: V * Lambda ---")
    print_matrix(VLambda)
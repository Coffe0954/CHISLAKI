def read_input_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    n = int(lines[0])
    A = []
    b = []
    for i in range(1, n + 1):
        row = list(map(float, lines[i].split()))
        A.append(row[:-1])
        b.append(row[-1])
    return n, A, b

def print_matrix(matrix):
    for row in matrix:
        print("  ".join(f"{val:10.4f}" for val in row))
    print()

def lu_decomposition_with_pivoting(A):
    n = len(A)
    U = [row[:] for row in A]
    L = [[0.0] * n for _ in range(n)]
    P = list(range(n))

    for i in range(n):
        max_val = abs(U[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(U[k][i]) > max_val:
                max_val = abs(U[k][i])
                max_row = k

        if max_val < 1e-12:
            raise ValueError("Matrix is singular.")

        if max_row != i:
            U[i], U[max_row] = U[max_row], U[i]
            P[i], P[max_row] = P[max_row], P[i]
            for k in range(i):
                L[i][k], L[max_row][k] = L[max_row][k], L[i][k]

        L[i][i] = 1.0
        for j in range(i, n):
            s = sum(L[i][k] * U[k][j] for k in range(i))
            U[i][j] = U[i][j] - s

        for j in range(i + 1, n):
            s = sum(L[j][k] * U[k][i] for k in range(i))
            L[j][i] = (U[j][i] - s) / U[i][i]
            U[j][i] = 0.0

    return L, U, P

def solve_lu(L, U, P, b):
    n = len(b)
    bp = [b[P[i]] for i in range(n)]

    z = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * z[j] for j in range(i))
        z[i] = bp[i] - s

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (z[i] - s) / U[i][i]

    return x

def calculate_determinant(U, P):
    det = 1.0
    n = len(U)
    swaps = 0
    temp_P = P[:]
    for i in range(n):
        while temp_P[i] != i:
            idx = temp_P[i]
            temp_P[i], temp_P[idx] = temp_P[idx], temp_P[i]
            swaps += 1

    sign = -1.0 if swaps % 2 != 0 else 1.0
    
    for i in range(n):
        det *= U[i][i]
        
    return sign * det

def invert_matrix(A, L, U, P):
    n = len(A)
    inv_A = [[0.0] * n for _ in range(n)]
    
    for j in range(n):
        e_col = [1.0 if i == j else 0.0 for i in range(n)]
        x_col = solve_lu(L, U, P, e_col)
        for i in range(n):
            inv_A[i][j] = x_col[i]
            
    return inv_A

def multiply_matrices(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    
    res = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            res[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    return res

def permutation_matrix_to_dense(P):
    n = len(P)
    Mat_P = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Mat_P[i][P[i]] = 1.0
    return Mat_P

if __name__ == "__main__":
    filename = "input_lr1_1.txt"
    n, A, b = read_input_file(filename)

    print("=== LAB 1.1 ===")
    print_matrix(A)
    print(f"b: {b}\n")

    L, U, P = lu_decomposition_with_pivoting(A)

    print_matrix(L)
    print_matrix(U)
    
    Mat_P = permutation_matrix_to_dense(P)
    print_matrix(Mat_P)

    x = solve_lu(L, U, P, b)
    print("--- X ---")
    for idx, val in enumerate(x, 1):
        print(f"x_{idx} = {val:.6f}")
    print()

    det_A = calculate_determinant(U, P)
    print(f"det(A) = {det_A:.6f}\n")

    inv_A = invert_matrix(A, L, U, P)
    print_matrix(inv_A)

    LU = multiply_matrices(L, U)
    PA = multiply_matrices(Mat_P, A)
    print_matrix(LU)
    print_matrix(PA)

    E_check = multiply_matrices(A, inv_A)
    print_matrix(E_check)
def simple_iteration(A, b, eps=0.01):
    n = len(b)
    x = [0.0] * n
    iterations = 0
    
    while True:
        x_new = [0.0] * n
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s += A[i][j] * x[j]
            x_new[i] = (b[i] - s) / A[i][i]
        
        iterations += 1
        max_diff = max(abs(x_new[i] - x[i]) for i in range(n))
        x = x_new[:]
        
        if max_diff < eps:
            break
            
    return x, iterations

def seidel_method(A, b, eps=0.01):
    n = len(b)
    x = [0.0] * n
    iterations = 0
    
    while True:
        x_new = x[:]
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        
        iterations += 1
        max_diff = max(abs(x_new[i] - x[i]) for i in range(n))
        
        if max_diff < eps:
            break
            
        x = x_new[:]
        
    return x, iterations

if __name__ == "__main__":
    A = [
        [21.0, -6.0, -9.0, -4.0],
        [-6.0, 20.0, -4.0, 2.0],
        [-2.0, -7.0, -20.0, 3.0],
        [4.0, 9.0, 6.0, 24.0]
    ]
    b = [127.0, -144.0, 236.0, -5.0]

    print("=== LAB 1.3 ===")
    print("Tochnost (epsilon) = 0.01\n")

    x_si, it_si = simple_iteration(A, b, 0.01)
    print(f"--- Simple Iteration (Iterations: {it_si}) ---")
    for idx, val in enumerate(x_si, 1):
        print(f"x_{idx} = {val:.6f}")
    print()

    x_se, it_se = seidel_method(A, b, 0.01)
    print(f"--- Seidel Method (Iterations: {it_se}) ---")
    for idx, val in enumerate(x_se, 1):
        print(f"x_{idx} = {val:.6f}")
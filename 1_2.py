def read_input_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    n = int(lines[0])
    a = list(map(float, lines[1].split()))
    b = list(map(float, lines[2].split()))
    c = list(map(float, lines[3].split()))
    d = list(map(float, lines[4].split()))
    return n, a, b, c, d

def tridiagonal_solve(n, a, b, c, d):
    P = [0.0] * n
    Q = [0.0] * n
    
    #прямой ход прогонки
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]
    
    for i in range(1, n - 1):
        denominator = b[i] + a[i] * P[i - 1]
        P[i] = -c[i] / denominator
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denominator
        
    #последний элемент (c[n-1] = 0)
    denominator = b[n - 1] + a[n - 1] * P[n - 2]
    Q[n - 1] = (d[n - 1] - a[n - 1] * Q[n - 2]) / denominator
    P[n - 1] = 0.0

    #обратный ход прогонки
    x = [0.0] * n
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]
        
    return P, Q, x

if __name__ == "__main__":
    filename = "input_lr1_2.txt"
    n, a, b, c, d = read_input_file(filename)

    print("=== LAB 1.2 ===")
    
    P, Q, x = tridiagonal_solve(n, a, b, c, d)

    print("--- Progonka coefficients P ---")
    for idx, val in enumerate(P, 1):
        print(f"P_{idx} = {val:.6f}")
    print()

    print("--- Progonka coefficients Q ---")
    for idx, val in enumerate(Q, 1):
        print(f"Q_{idx} = {val:.6f}")
    print()

    print("--- Solution X ---")
    for idx, val in enumerate(x, 1):
        print(f"x_{idx} = {val:.6f}")
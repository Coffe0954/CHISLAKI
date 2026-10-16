import numpy as np

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
    return n, np.array(A)

def qr_decomposition_gram_schmidt(A):
    n = A.shape[0]
    Q = np.zeros((n, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        v = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        if R[j, j] > 1e-12:
            Q[:, j] = v / R[j, j]
        else:
            Q[:, j] = 0.0
            
    return Q, R

if __name__ == "__main__":
    filename = "input_lr1_5.txt"
    n, A = read_input_file(filename)

    print("=== LAB 1.5 (QR-razlozhenie) ===")
    print("Ishodnaya matritsa A:")
    print(np.round(A, 4))
    print()

    Q, R = qr_decomposition_gram_schmidt(A)

    print("Matritsa Q:")
    print(np.round(Q, 4))
    print()

    print("Matritsa R:")
    print(np.round(R, 4))
    print()

    QR_check = np.dot(Q, R)
    print("Proverka: Q * R (dolzhna byt ishodnaya matritsa A):")
    print(np.round(QR_check, 4))
    print()

    eigenvalues = np.linalg.eigvals(A)
    print("Sobstvennye znacheniya matritsy (vklyuchaya kompleksno-sopryazhennye):")
    for idx, val in enumerate(eigenvalues, 1):
        print(f"lambda_{idx} = {val}")
import math
import numpy as np

# Lab 2.2, Variant 16 (a = 2):
# f1(x1, x2) = 2*x1 - cos(x2) = 0
# f2(x1, x2) = 2*x2 - exp(x1) = 0

def system_F(X):
    x1, x2 = X
    f1 = 2.0 * x1 - math.cos(x2)
    f2 = 2.0 * x2 - math.exp(x1)
    return np.array([f1, f2], dtype=float)

def phi(X):
    x1, x2 = X
    new_x1 = math.cos(x2) / 2.0
    new_x2 = math.exp(x1) / 2.0
    return np.array([new_x1, new_x2], dtype=float)

def jacobian(X):
    x1, x2 = X
    J = np.array([
        [2.0, math.sin(x2)],
        [-math.exp(x1), 2.0]
    ], dtype=float)
    return J

def simple_iteration_system(X0, eps=1e-5, max_iter=100):
    X = np.array(X0, dtype=float)
    for it in range(1, max_iter + 1):
        X_new = phi(X)
        if np.linalg.norm(X_new - X, ord=np.inf) < eps:
            return X_new, it
        X = X_new
    raise ValueError("Simple iteration system method diverged")

def newton_system(X0, eps=1e-5, max_iter=100):
    X = np.array(X0, dtype=float)
    for it in range(1, max_iter + 1):
        F = system_F(X)
        if np.linalg.norm(F, ord=np.inf) < eps:
            return X, it - 1
            
        J = jacobian(X)
        try:
            deltaX = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            raise ZeroDivisionError("Jacobian matrix is singular")
            
        X_new = X + deltaX
        
        if np.linalg.norm(deltaX, ord=np.inf) < eps:
            return X_new, it
            
        X = X_new
    raise ValueError("Newton's system method diverged")

if __name__ == "__main__":
    X0 = [0.5, 0.5]
    
    print("=== LAB 2.2 (Variant 16) ===")
    print(f"Initial approximation: X0 = {X0}\n")
    
    try:
        root_si, it_si = simple_iteration_system(X0)
        print("--- Simple Iteration Method (System) ---")
        print(f"Root: x1 = {root_si[0]:.6f}, x2 = {root_si[1]:.6f}")
        print(f"Iterations count: {it_si}\n")
    except Exception as e:
        print(f"Simple iteration system failed: {e}\n")
        
    try:
        root_n, it_n = newton_system(X0)
        print("--- Newton's Method (System) ---")
        print(f"Root: x1 = {root_n[0]:.6f}, x2 = {root_n[1]:.6f}")
        print(f"Iterations count: {it_n}\n")
    except Exception as e:
        print(f"Newton's system method failed: {e}\n")
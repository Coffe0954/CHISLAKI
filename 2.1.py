import math

# Lab 2.1, Variant 16: x * exp(x) + x^2 - 1 = 0
def f(x):
    return x * math.exp(x) + x**2 - 1.0

def df(x):
    return math.exp(x) * (1.0 + x) + 2.0 * x

def phi(x):
    return (1.0 - x**2) / math.exp(x)

def simple_iteration_scalar(x0, eps=1e-5, max_iter=100):
    x = x0
    for it in range(1, max_iter + 1):
        x_new = phi(x)
        if abs(x_new - x) < eps:
            return x_new, it
        x = x_new
    raise ValueError("Simple iteration method diverged or exceeded max iterations")

def newton_scalar(x0, eps=1e-5, max_iter=100):
    x = x0
    for it in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-12:
            raise ZeroDivisionError("Derivative is too close to zero")
        x_new = x - fx / dfx
        if abs(x_new - x) < eps:
            return x_new, it
        x = x_new
    raise ValueError("Newton's method diverged")

if __name__ == "__main__":
    x0 = 0.5
    print("=== LAB 2.1 (Variant 16) ===")
    print(f"Initial guess: x0 = {x0}\n")
    
    try:
        root_si, it_si = simple_iteration_scalar(x0)
        print("--- Simple Iteration Method ---")
        print(f"Root: x = {root_si:.6f}")
        print(f"Iterations count: {it_si}\n")
    except Exception as e:
        print(f"Simple iteration failed: {e}\n")
        
    try:
        root_n, it_n = newton_scalar(x0)
        print("--- Newton's Method ---")
        print(f"Root: x = {root_n:.6f}")
        print(f"Iterations count: {it_n}\n")
    except Exception as e:
        print(f"Newton's method failed: {e}\n")
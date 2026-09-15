import os
import sys

THETAS_PATH = "thetas.csv"


def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)


def load_thetas(path=THETAS_PATH):
    # Before training theta0 and theta1 are 0.
    try:
        with open(path) as f:
            theta0, theta1 = f.read().split(",")
            return float(theta0), float(theta1)
    except (FileNotFoundError, ValueError):
        return 0.0, 0.0


def ask_mileage():
    # Keep asking until the input is a valid number (int or float).
    while True:
        raw = input("Enter a mileage (km): ").strip()
        try:
            mileage = float(raw)
        except ValueError:
            print("Invalid input: please enter an integer or a float.")
            continue
        if mileage != mileage or mileage in (float("inf"), float("-inf")):
            print("Invalid input: please enter a finite number.")
            continue
        if mileage < 0:
            print("Invalid input: a mileage cannot be negative.")
            continue
        return mileage


def main():
    if not os.path.isfile(THETAS_PATH):
        print(f"No trained model found ({THETAS_PATH} is missing).")
        print("Run 'python train.py' first.")
        sys.exit(1)

    theta0, theta1 = load_thetas()
    try:
        mileage = ask_mileage()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit(1)
    # A price can't be negative, so floor the result at 0.
    price = max(0.0, estimate_price(mileage, theta0, theta1))
    print(f"Estimated price: {price:.2f}")


if __name__ == "__main__":
    main()

from predict import estimate_price

LEARNING_RATE = 0.5
ITERATIONS = 10000


def load_data(path="data.csv"):
    mileages, prices = [], []
    with open(path) as f:
        next(f)  # skip the "km,price" header
        for line in f:
            line = line.strip()
            if not line:
                continue
            km, price = line.split(",")
            mileages.append(float(km))
            prices.append(float(price))
    return mileages, prices


def train(mileages, prices):
    m = len(mileages)

    # Normalize the mileage so gradient descent converges (km values are huge).
    x_min, x_max = min(mileages), max(mileages)
    x_norm = [(km - x_min) / (x_max - x_min) for km in mileages]

    theta0, theta1 = 0.0, 0.0
    for _ in range(ITERATIONS):
        errors = [estimate_price(x_norm[i], theta0, theta1) - prices[i]
                  for i in range(m)]
        tmp0 = LEARNING_RATE * sum(errors) / m
        tmp1 = LEARNING_RATE * sum(errors[i] * x_norm[i] for i in range(m)) / m
        theta0 -= tmp0
        theta1 -= tmp1

    # De-normalize so predict.py can use raw mileage values.
    theta1 = theta1 / (x_max - x_min)
    theta0 = theta0 - theta1 * x_min
    return theta0, theta1


def save_thetas(theta0, theta1, path="thetas.csv"):
    with open(path, "w") as f:
        f.write(f"{theta0},{theta1}")


def main():
    mileages, prices = load_data()
    theta0, theta1 = train(mileages, prices)
    save_thetas(theta0, theta1)
    print(f"Training done. theta0 = {theta0:.6f}, theta1 = {theta1:.6f}")


if __name__ == "__main__":
    main()

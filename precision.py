from predict import estimate_price, load_thetas
from train import load_data


def main():
    mileages, prices = load_data()
    theta0, theta1 = load_thetas()
    m = len(prices)

    mean_price = sum(prices) / m
    predictions = [estimate_price(mileages[i], theta0, theta1) for i in range(m)]

    # Sum of squared errors and total sum of squares.
    ss_res = sum((prices[i] - predictions[i]) ** 2 for i in range(m))
    ss_tot = sum((prices[i] - mean_price) ** 2 for i in range(m))

    r2 = 1 - ss_res / ss_tot          # 1.0 = perfect fit
    rmse = (ss_res / m) ** 0.5        # average error in price units

    print(f"R2   = {r2:.4f}")
    print(f"RMSE = {rmse:.2f}")


if __name__ == "__main__":
    main()

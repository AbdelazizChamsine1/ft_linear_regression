import matplotlib.pyplot as plt

from predict import estimate_price, load_thetas
from train import load_data


def main():
    mileages, prices = load_data()
    theta0, theta1 = load_thetas()

    # The data distribution.
    plt.scatter(mileages, prices, color="blue", label="Data")

    # The regression line on the same graph.
    line_x = [min(mileages), max(mileages)]
    line_y = [estimate_price(x, theta0, theta1) for x in line_x]
    plt.plot(line_x, line_y, color="red", label="Regression line")

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.legend()
    plt.title("Car price vs mileage")
    plt.show()


if __name__ == "__main__":
    main()

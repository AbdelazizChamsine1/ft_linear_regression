# ft_linear_regression

A machine learning project: predicting a car's price from its mileage
using a simple linear regression trained with gradient descent.

The model is just a line:

```
price = theta0 + theta1 * mileage
```

Training finds good values for `theta0` and `theta1` from the data.

## Files

- `data.csv` - the dataset (mileage, price)
- `train.py` - trains the model and saves the thetas to `thetas.csv`
- `predict.py` - asks you for a mileage and gives back an estimated price
- `plot.py` - shows the data and the regression line on a graph (bonus)
- `precision.py` - prints how accurate the model is (bonus)

## How to use

Train the model first:

```
python train.py
```

This reads `data.csv`, runs gradient descent, and writes `theta0` and
`theta1` into `thetas.csv`.

Then make a prediction:

```
python predict.py
```

It will prompt you for a mileage and print the estimated price. If you run it
before training, both thetas are 0, so it just returns 0.

## Bonus

```
python plot.py        # scatter plot of the data + the fitted line
python precision.py   # R2 and RMSE of the model
```

## A couple of notes

- The mileage values are huge (tens of thousands) compared to the prices, which
  makes gradient descent blow up. So during training I normalize the mileage to
  a 0–1 range, and once it's done I convert the thetas back to real units before
  saving. That way `predict.py` works with normal mileage values.
- A straight line will eventually predict negative prices for very high mileage
  (it's just extrapolating past the data). Since a price can't be negative,
  `predict.py` floors the result at 0.

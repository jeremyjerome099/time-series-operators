import numpy as np


def ts_decay_linear(x, d, dense=False):
 """
    Compute the linear decay (weighted average) of the last d days of a time series.

    - x: sequence-like, ordered from most recent (Day 0) to oldest (Day -(d-1)).
    - d: positive integer, number of days to use in the decay.
    - dense: bool
        - False (sparse mode): treat NaN as 0 during computation.
        - True (dense mode): do not treat NaN as 0; NaNs will propagate (result will be NaN if any NaN among the first d values).

    Returns:
        float: the weighted linear decay value.

    Notes:
        - If x has fewer than d entries, missing days are treated as 0.
        - Weights are [d, d-1, ..., 1]; Denominator = d*(d+1)/2.
        - Example: x = [30, 5, 4, 5, 6], d = 5 -> 198 / 15 = 13.2
    """
    if d <= 0:
        raise ValueError("d must be a positive integer.")
    x_arr = np.asarray(x, dtype=float)
    d = int(d)

    weights = np.arange(d, 0, -1, dtype=float)

    values = np.zeros(d, dtype=float)
    upto = min(d, x_arr.size)
    if upto > 0:
        segment = x_arr[:upto].astype(float)
        if not dense:
            segment = np.where(np.isnan(segment), 0.0, segment)
        values[:upto] = segment

    numerator = float(np.dot(values, weights))
    denominator = d * (d + 1) / 2.0

    return numerator / denominator

if __name__ == "__main__":
    print("This program computes the linear decay (weighted average) over the last five days.")
    print("Please enter the stock prices for Day 0 (today) through Day -4, separated by spaces.")
    while True:
        user_input = input("Enter 5 numbers (Day0 Day-1 Day-2 Day-3 Day-4): ").strip()
        parts = user_input.split()
        if len(parts) != 5:
            print("Error: You must provide exactly 5 numbers. Example: 30 5 4 5 6")
            continue
        try:
            prices = [float(p) for p in parts]
        except ValueError:
            print("Error: All values must be valid numbers.")
            continue
        break

    result = ts_decay_linear(prices, d=5, dense=False)
    print(f"Weighted linear decay value for the last five days: {result}")
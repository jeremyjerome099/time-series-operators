import numpy as np


def trade_when(trigger_trade, alpha_exp, trigger_exit, initial_alpha=np.nan):
    """
    Compute the Alpha series according to the Trade_When operator.

    - trigger_trade: sequence (x) of trade signals (positive to trade)
    - alpha_exp: sequence (y) of Alpha values to apply when trading
    - trigger_exit: sequence (z) of exit signals (positive to exit -> Alpha = NaN)
    - initial_alpha: starting previous Alpha (default NaN)

    Returns:
        numpy.ndarray: The resulting Alpha series
    """
    x = np.asarray(trigger_trade, dtype=float)
    a = np.asarray(alpha_exp, dtype=float)
    z = np.asarray(trigger_exit, dtype=float)

    if not (x.size == a.size == z.size):
        raise ValueError("All inputs must be the same length.")

    n = x.size
    out = np.empty(n, dtype=float)
    prev = initial_alpha

    for i in range(n):
        if z[i] > 0:
            out[i] = np.nan
            prev = out[i]
        elif x[i] > 0:
            out[i] = a[i]
            prev = out[i]
        else:
            out[i] = prev

    return out


# Example usage
if __name__ == "__main__":
    # Example sequences
    trigger_trade = [0, 1, 0, 0, 1, 0]  # x
    alpha_exp = [0.10, 0.75, -0.20, 0.30, 0.40, 0.90]  # y
    trigger_exit = [0, 0, 0, 1, 0, 0]  # z

    initial_alpha = 0.0  # starting previous Alpha

    result = trade_when(trigger_trade, alpha_exp, trigger_exit, initial_alpha)
    print("Trade_When Alpha series:")
    print(result)
    # Expected reasoning:
    # t0: exit0, trade0 -> Alpha = initial_alpha = 0.0
    # t1: exit0, trade1 -> Alpha = alpha_exp[1] = 0.75
    # t2: exit0, trade0 -> Alpha = prev = 0.75
    # t3: exit1 (>0) -> Alpha = NaN
    # t4: exit0, trade1 -> Alpha = alpha_exp[4] = 0.40
    # t5: exit0, trade0 -> Alpha = prev = 0.40

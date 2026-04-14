"""
IBM Cloud Code Engine Demo – 5 Parallel Universes
===================================================
Everyone texts a monthly savings amount. Code Engine spins up
5 parallel workers — each simulates a different "universe" for
what happens to that money over 10 years.

Worker 0: The Gambler      – blows it at the roulette table every month
Worker 1: The Investor     – dollar-cost averages into the S&P 500
Worker 2: The Lotto Addict – buys scratch tickets every month
Worker 3: The Mattress     – stuffs cash under the bed (inflation eats it)
Worker 4: The Crypto King  – dollar-cost averages into crypto

Environment variables:
  MONTHLY_AMOUNT  – dollar amount per month (default $500, capped at $5,000)
  TWILIO_TOKEN
  PHONE_NUMBER
  JOB_INDEX       – set automatically by Code Engine (0-4)
"""

import os
import numpy as np
from twilio.rest import Client


# ── Configuration ────────────────────────────────────────────────────
NUM_SIMULATIONS = 10_000
MONTHS = 120              # 10 years
MAX_MONTHLY = 5_000

STRATEGIES = [
    "The Gambler",
    "The Investor",
    "The Lotto Addict",
    "The Mattress",
    "The Crypto King",
]


# ── Strategy Simulators ─────────────────────────────────────────────

def sim_gambler(monthly: float) -> np.ndarray:
    """
    The Gambler – Every month, walks into the casino and bets the entire
    amount on red at roulette. Win (18/38 ≈ 47.4%) = keeps the winnings.
    Lose = $0 that month. Saves whatever comes out.
    """
    p_win = 18 / 38  # American roulette
    # Each month is a coin flip: win = 2x, lose = 0
    wins = np.random.random((NUM_SIMULATIONS, MONTHS)) < p_win
    monthly_outcome = wins.astype(float) * (monthly * 2)  # win pays 2x
    cumulative = np.cumsum(monthly_outcome, axis=1)
    return cumulative


def sim_investor(monthly: float) -> np.ndarray:
    """
    The Investor – Dollar-cost averages into the S&P 500 every month.
    Uses historical parameters: ~10% annual return, ~15% annual volatility.
    """
    annual_return = 0.10
    annual_vol = 0.15
    monthly_mu = annual_return / 12
    monthly_sigma = annual_vol / np.sqrt(12)

    # Simulate month-by-month portfolio growth with DCA
    returns = np.random.normal(monthly_mu, monthly_sigma,
                               (NUM_SIMULATIONS, MONTHS))
    portfolios = np.zeros((NUM_SIMULATIONS, MONTHS))

    for m in range(MONTHS):
        if m == 0:
            portfolios[:, m] = monthly * (1 + returns[:, m])
        else:
            portfolios[:, m] = (portfolios[:, m - 1] + monthly) * (1 + returns[:, m])

    return portfolios


def sim_lotto(monthly: float) -> np.ndarray:
    """
    The Lotto Addict – Spends the full amount on scratch tickets each month.
    Modeled as: 60% chance of nothing, 25% chance win 0.5x back,
    10% chance win 2x, 4% chance win 5x, 0.9% chance win 20x,
    0.1% chance win 100x (the big story).
    Expected return ≈ $0.59 per dollar. The house always wins.
    """
    thresholds = np.array([0.60, 0.85, 0.95, 0.99, 0.999, 1.0])
    payouts = np.array([0.0, 0.5, 2.0, 5.0, 20.0, 100.0])

    rolls = np.random.random((NUM_SIMULATIONS, MONTHS))
    monthly_winnings = np.zeros_like(rolls)

    for i in range(len(thresholds)):
        if i == 0:
            mask = rolls < thresholds[i]
        else:
            mask = (rolls >= thresholds[i - 1]) & (rolls < thresholds[i])
        monthly_winnings[mask] = payouts[i] * monthly

    cumulative = np.cumsum(monthly_winnings, axis=1)
    return cumulative


def sim_mattress(monthly: float) -> np.ndarray:
    """
    The Mattress – Stuffs cash under the bed every month.
    Nominal balance grows linearly, but we show real purchasing power
    eroded by ~3.5% annual inflation (with noise).
    """
    annual_inflation = 0.035
    monthly_inf = annual_inflation / 12
    inf_vol = 0.005  # small monthly noise

    inflation = np.random.normal(monthly_inf, inf_vol,
                                 (NUM_SIMULATIONS, MONTHS))

    # Track real purchasing power
    portfolios = np.zeros((NUM_SIMULATIONS, MONTHS))
    cum_inflation = np.ones(NUM_SIMULATIONS)

    for m in range(MONTHS):
        cum_inflation *= (1 + inflation[:, m])
        nominal = monthly * (m + 1)
        portfolios[:, m] = nominal / cum_inflation  # real value

    return portfolios


def sim_crypto(monthly: float) -> np.ndarray:
    """
    The Crypto King – Dollar-cost averages into crypto every month.
    Wild ride: ~25% annual return, ~80% annual volatility.
    You might be a millionaire. You might have bus fare.
    """
    annual_return = 0.25
    annual_vol = 0.80
    monthly_mu = annual_return / 12
    monthly_sigma = annual_vol / np.sqrt(12)

    returns = np.random.normal(monthly_mu, monthly_sigma,
                               (NUM_SIMULATIONS, MONTHS))
    portfolios = np.zeros((NUM_SIMULATIONS, MONTHS))

    for m in range(MONTHS):
        if m == 0:
            portfolios[:, m] = monthly * (1 + returns[:, m])
        else:
            portfolios[:, m] = (portfolios[:, m - 1] + monthly) * (1 + returns[:, m])

    return portfolios


SIMULATORS = [sim_gambler, sim_investor, sim_lotto, sim_mattress, sim_crypto]


# ── Analysis ─────────────────────────────────────────────────────────

def analyze(portfolios: np.ndarray, monthly: float) -> dict:
    """Compute summary stats from the simulation results."""
    total_contributed = monthly * MONTHS
    final_values = portfolios[:, -1]

    return {
        "contributed":  total_contributed,
        "mean":         float(np.mean(final_values)),
        "median":       float(np.median(final_values)),
        "best":         float(np.percentile(final_values, 95)),
        "worst":        float(np.percentile(final_values, 5)),
        "prob_profit":  float(np.mean(final_values > total_contributed) * 100),
        "prob_double":  float(np.mean(final_values > total_contributed * 2) * 100),
    }


# ── SMS Formatting ───────────────────────────────────────────────────

TAGLINES = {
    "The Gambler":      "You walked into the casino every month for 10 years.",
    "The Investor":     "You bought the S&P 500 like clockwork for 10 years.",
    "The Lotto Addict": "You bought scratch tickets every month for 10 years.",
    "The Mattress":     "You hid cash under your mattress for 10 years.",
    "The Crypto King":  "You aped into crypto every month for 10 years.",
}

FLAVOR = {
    "The Gambler":      "The house always wins. But hey, free drinks.",
    "The Investor":     "Boring works. Warren Buffett sends his regards.",
    "The Lotto Addict": "You could've bought a boat. Instead you bought hope.",
    "The Mattress":     "Safe? Sure. But inflation doesn't sleep either.",
    "The Crypto King":  "To the moon — or to zero. No in-between energy.",
}


def format_sms(strategy: str, idx: int, monthly: float, stats: dict) -> str:
    """Build the SMS body for one universe."""
    tagline = TAGLINES[strategy]
    flavor = FLAVOR[strategy]

    contributed = stats["contributed"]
    mean_val = stats["mean"]
    median_val = stats["median"]
    gain_pct = ((median_val - contributed) / contributed) * 100

    return (
        f"Simulation: {strategy}\n"
        f"----\n"
        f"{tagline}\n"
        f"\n"
        f"You put in: ${contributed:,.0f} (${monthly:,.0f}/mo × 10 yrs)\n"
        f"\n"
        f"Median outcome:  ${median_val:,.0f} ({gain_pct:+.1f}%)\n"
        f"Average outcome: ${mean_val:,.0f}\n"
        f"Best case (95th): ${stats['best']:,.0f}\n"
        f"Worst case (5th): ${stats['worst']:,.0f}\n"
        f"\n"
        f"Chance you come out ahead: {stats['prob_profit']:.0f}%\n"
        f"Chance you double your money: {stats['prob_double']:.0f}%\n"
        f"\n"
        f"{flavor}\n"
        f"\n"
        f"[{NUM_SIMULATIONS:,} simulations | Universe {idx + 1} of 5]"
    )


# ── Main ─────────────────────────────────────────────────────────────

def main():
    job_index = int(os.getenv("JOB_INDEX", "0"))

    if job_index >= len(STRATEGIES):
        print(f"Worker {job_index}: no strategy assigned. Exiting.")
        return

    strategy = STRATEGIES[job_index]
    monthly = min(float(os.environ.get("MONTHLY_AMOUNT", "500")), MAX_MONTHLY)

    # ── Twilio setup ─────────────────────────────────────────────
    account_sid = "AC9c4c7cc858805c2d3661e38214d8e505"
    auth_token  = os.environ["TWILIO_TOKEN"]
    from_number = "+18674571410"
    to_number   = os.environ["PHONE_NUMBER"]
    twilio_client = Client(account_sid, auth_token)

    # ── Simulate ─────────────────────────────────────────────────
    print(f"Worker {job_index}: simulating '{strategy}' "
          f"(${monthly:,.0f}/mo, {NUM_SIMULATIONS:,} sims)...")

    simulator = SIMULATORS[job_index]
    portfolios = simulator(monthly)
    stats = analyze(portfolios, monthly)

    # ── Send SMS ─────────────────────────────────────────────────
    body = format_sms(strategy, job_index, monthly, stats)
    message = twilio_client.messages.create(
        body=body,
        from_=from_number,
        to=to_number,
    )

    print(f"Worker {job_index} ({strategy}): SMS sent, SID={message.sid}")
    print(body)


if __name__ == "__main__":
    main()

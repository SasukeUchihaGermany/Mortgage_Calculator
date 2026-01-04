"""
UK Buy-to-Let Investment Model (Ltd-owned, realism-first)
--------------------------------------------------------
Outputs: profit if sold at end of each year 1..N, plus IRR, DSCR, etc.
Includes:
- House price path + sale event
- Mortgage amortization + equity build
- Upfront + exit transaction costs (incl SDLT model)
- Ltd-owned taxation (corp tax on rental profits + corp tax on chargeable gain)
- Voids/bad debt, letting fees, capex reserve + maintenance shocks
- Inflation growth on nominal costs + real discounting (year-by-year)
- Optional Monte Carlo (distributions defined using median + 95% ranges)

Notes:
- UK tax is complex and changes. This is a model, not tax advice.
- Marginal relief corporation tax is approximated unless you plug a precise calculator.
- “Enveloped dwellings” SDLT 17% for certain corporate purchases >£500k exists; rental business relief may apply. (Model uses a switch.)
- ATED may apply for company-owned residential property above thresholds; often relief is available for property rental businesses, but filings can still exist. (Model provides a switch.)

Sources:
- Corporation tax bands & marginal relief concept: GOV.UK. (19% <50k; 25% >250k) https://www.gov.uk/guidance/corporation-tax-marginal-relief
- Corporation tax rates: GOV.UK “rates and allowances”; PwC summary also confirms continuation into FY starting 1 Apr 2026.
- SDLT residential rates: GOV.UK
- SDLT corporate bodies 17% rule: GOV.UK
- ATED basics: GOV.UK
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import math
import random

try:
    import numpy as np
except ImportError:
    np = None


# ----------------------------
# Helpers: distributions
# ----------------------------

def triangular_from_95(median: float, lo95: float, hi95: float, rng: random.Random) -> float:
    """
    Triangular distribution where we treat lo95 and hi95 as approx 2.5% and 97.5%.
    We back out a triangular (a, c, b) by setting:
      a=lo95, b=hi95, c=median
    This is NOT a perfect mapping from quantiles -> triangle, but it's transparent and stable.
    """
    a, b, c = lo95, hi95, median
    return rng.triangular(a, b, c)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ----------------------------
# SDLT models (England/NI)
# ----------------------------

def sdlt_residential_standard(price: float) -> float:
    """
    Standard SDLT bands (England/NI): GOV.UK. 0% up to 125k, 2% 125-250, 5% 250-925, 10% 925-1.5m, 12% above.
    (If you need post-Apr-2025 changes or surcharges, keep using the explicit knobs below.)
    """
    bands = [
        (125_000, 0.00),
        (250_000, 0.02),
        (925_000, 0.05),
        (1_500_000, 0.10),
        (float("inf"), 0.12),
    ]
    return _sdlt_progressive(price, bands)

def sdlt_residential_higher_rates_additional_property(price: float, surcharge_pp: float = 0.05) -> float:
    """
    Higher rates for additional properties are standard rates + a surcharge percentage points.
    Surcharge has changed over time; model it as a knob.
    """
    base = sdlt_residential_standard(price)
    return base + price * surcharge_pp

def sdlt_corporate_enveloped_17pct(price: float) -> float:
    """
    Corporate bodies ‘enveloped dwellings’ rate (17%) for certain corporate buyers for residential > £500k (GOV.UK).
    In reality: reliefs may apply (e.g., qualifying rental business), so this is a switchable worst-case.
    """
    if price <= 500_000:
        return 0.0
    return 0.17 * price

def _sdlt_progressive(price: float, bands: List[Tuple[float, float]]) -> float:
    tax = 0.0
    prev = 0.0
    for upper, rate in bands:
        slab = min(price, upper) - prev
        if slab > 0:
            tax += slab * rate
        prev = upper
        if price <= upper:
            break
    return tax


# ----------------------------
# Mortgage math
# ----------------------------

def monthly_payment_repayment(loan: float, annual_rate_pct: float, term_years: int) -> float:
    r = (annual_rate_pct / 100.0) / 12.0
    n = term_years * 12
    if annual_rate_pct <= 0:
        return loan / n
    return loan * r / (1 - (1 + r) ** (-n))

def amortization_year(
    balance_start: float,
    annual_rate_pct: float,
    monthly_payment: float,
    months: int = 12
) -> Tuple[float, float, float]:
    """
    Returns (balance_end, interest_paid, principal_paid) over 'months'.
    """
    r = (annual_rate_pct / 100.0) / 12.0
    bal = balance_start
    interest = 0.0
    principal = 0.0
    for _ in range(months):
        i = bal * r
        p = monthly_payment - i
        if p < 0:
            # Payment doesn't cover interest -> negative amortization.
            # For realism, we still book it; bal increases.
            p = 0.0
            bal = bal + i - monthly_payment
            interest += monthly_payment
        else:
            bal = max(0.0, bal - p)
            interest += i
            principal += p
        if bal <= 0:
            break
    return bal, interest, principal


# ----------------------------
# Corp tax (Ltd) approximation
# ----------------------------

def corp_tax_on_profit(profit: float) -> float:
    """
    Verified bands: 19% small profits <= 50k, 25% >= 250k, marginal relief between. GOV.UK.
    Exact marginal relief requires associated companies + accounting period details; we approximate linearly
    between 19% and 25% for transparency.
    """
    if profit <= 0:
        return 0.0
    lo, hi = 50_000.0, 250_000.0
    if profit <= lo:
        return 0.19 * profit
    if profit >= hi:
        return 0.25 * profit
    eff_rate = 0.19 + (profit - lo) / (hi - lo) * (0.25 - 0.19)  # approximation
    return eff_rate * profit


# ----------------------------
# Model parameters
# ----------------------------

@dataclass
class Params:
    # Purchase
    price: float = 300_000.0                 # median (£250k–£450k)
    deposit_pct: float = 0.25                # median (0.20–0.40)

    # Mortgage
    term_years: int = 25
    mortgage_type: str = "repayment"         # "repayment" or "interest_only"
    interest_only_balloon: bool = True       # if interest-only: balloon repayment at sale unless refinanced

    # Income
    rent_monthly_start: float = 1500.0       # median (£1200–£1900)
    rent_growth_pct: float = 3.0             # median (0–6)

    # Operating costs (starting, nominal per month)
    opex_monthly_start: float = 260.0        # median (£180–£450)
    council_tax_landlord_monthly: float = 0.0 # usually tenant pays, but set if HMO/voids/etc.
    insurance_monthly: float = 25.0
    ground_rent_service_charge_monthly: float = 0.0  # flats: can be large

    # Variable cost fractions of gross rent (annual)
    void_bad_debt_frac: float = 0.08         # median 8% (0–12)  [safety-biased]
    letting_mgmt_fee_frac: float = 0.12      # median 12% (0.08–0.15) full management typical range
    capex_reserve_frac: float = 0.06         # median 6% (0.02–0.12) [safety-biased]

    # Maintenance shocks (non-linear realism)
    maint_shock_prob_per_year: float = 0.12  # median 12% (5–25): chance of a “big hit” in a year
    maint_shock_size_months_rent: float = 1.0 # median 1 month rent (0.5–3.0): shock magnitude

    # Transactions
    buy_cost_pct: float = 0.008              # median 0.8% (0.3–1.5)
    buy_cost_fixed: float = 1800.0           # median £1800 (£500–£4000)
    sell_cost_pct: float = 0.015             # median 1.5% (1.0–2.5)
    sell_cost_fixed: float = 1500.0          # median £1500 (£500–£3000)

    # SDLT options
    apply_sdlt: bool = True
    sdlt_mode: str = "higher_additional"     # "standard", "higher_additional", "corp_17pct_worstcase"
    sdlt_surcharge_pp: float = 0.05          # 5% surcharge knob for higher rates (policy changes over time)

    # ATED (optional, often relieved but filings may exist)
    apply_ated: bool = False
    ated_annual_charge: float = 0.0          # set if you want to model net cost even if relief expected

    # Price/inflation paths
    inflation_pct: float = 3.5               # median (1–8)
    house_price_growth_pct: float = 3.0      # median (-2–6)

    # Interest rate path behaviour
    # If you provide interest_rate_path, it overrides the “flat” annual_rate_pct for those years.
    annual_rate_pct: float = 5.0             # used if no path
    interest_rate_path: Optional[List[float]] = None  # e.g. [4.5,4.5,5.5,6.0,...] per-year

    # Horizon
    max_years: int = 25
    sale_year: int = 10

    # Company-level: we apply corp tax on rental profits and chargeable gains.
    # We do NOT model dividend tax on extraction (huge in real life).
    include_gain_in_corp_tax: bool = True


# ----------------------------
# Core simulation
# ----------------------------

@dataclass
class YearResult:
    year: int
    rate_pct: float
    rent_gross: float
    rent_net_before_finance: float
    interest_paid: float
    principal_paid: float
    mortgage_paid: float
    corp_tax: float
    post_tax_cashflow: float
    balance_end: float
    property_value_end: float
    equity_end: float

def compute_sdlt(p: Params) -> float:
    if not p.apply_sdlt:
        return 0.0
    if p.sdlt_mode == "standard":
        return sdlt_residential_standard(p.price)
    if p.sdlt_mode == "higher_additional":
        return sdlt_residential_higher_rates_additional_property(p.price, p.sdlt_surcharge_pp)
    if p.sdlt_mode == "corp_17pct_worstcase":
        # Worst-case if relief does NOT apply (see GOV.UK corporate bodies SDLT guidance).
        return sdlt_corporate_enveloped_17pct(p.price)
    raise ValueError(f"Unknown sdlt_mode: {p.sdlt_mode}")

def simulate(p: Params, rng: Optional[random.Random] = None) -> Dict[str, object]:
    """
    Deterministic if rng is None (no maintenance shock randomness).
    If rng is provided, includes stochastic maintenance shocks.
    """
    loan0 = p.price * (1.0 - p.deposit_pct)
    deposit = p.price * p.deposit_pct

    sdlt = compute_sdlt(p)
    buy_costs = p.price * p.buy_cost_pct + p.buy_cost_fixed
    initial_cash_out = deposit + sdlt + buy_costs

    # Mortgage payment
    if p.mortgage_type == "repayment":
        monthly_pmt = monthly_payment_repayment(loan0, p.annual_rate_pct if not p.interest_rate_path else p.interest_rate_path[0], p.term_years)
    elif p.mortgage_type == "interest_only":
        # Interest-only: monthly payment = interest only, computed year-by-year in loop
        monthly_pmt = 0.0
    else:
        raise ValueError("mortgage_type must be 'repayment' or 'interest_only'")

    # State
    bal = loan0
    rent_m = p.rent_monthly_start
    opex_m = p.opex_monthly_start
    insurance_m = p.insurance_monthly
    council_m = p.council_tax_landlord_monthly
    service_m = p.ground_rent_service_charge_monthly

    results: List[YearResult] = []
    cashflows_nominal: List[float] = [-initial_cash_out]  # time 0
    cashflows_real: List[float] = [-initial_cash_out]     # time 0 (real == nominal at t0)

    for y in range(1, p.max_years + 1):
        rate = p.annual_rate_pct
        if p.interest_rate_path and y - 1 < len(p.interest_rate_path):
            rate = p.interest_rate_path[y - 1]

        # House price at end of year y
        prop_val = p.price * (1 + p.house_price_growth_pct / 100.0) ** y

        # Income
        gross_rent = rent_m * 12.0

        # Voids/bad debt
        void_loss = gross_rent * p.void_bad_debt_frac

        # Letting management fee
        mgmt_fee = gross_rent * p.letting_mgmt_fee_frac

        # Capex reserve (annual)
        capex_reserve = gross_rent * p.capex_reserve_frac

        # Maintenance shock (random, lumpy)
        shock = 0.0
        if rng is not None and rng.random() < p.maint_shock_prob_per_year:
            shock = p.maint_shock_size_months_rent * rent_m  # one-off nominal cost this year

        # Opex inflated with inflation each year (cost realism)
        # Costs are treated as paid during the year; we use end-of-year accounting for simplicity.
        opex = opex_m * 12.0
        insurance = insurance_m * 12.0
        council = council_m * 12.0
        service = service_m * 12.0

        rent_net_before_finance = gross_rent - void_loss - mgmt_fee - capex_reserve - shock - opex - insurance - council - service

        # Financing: mortgage cash paid, interest/principal
        interest_paid = 0.0
        principal_paid = 0.0
        mortgage_paid = 0.0

        if y <= p.term_years and bal > 0:
            if p.mortgage_type == "repayment":
                # Payment should update if rate path changes (realistic for variable/product expiry).
                monthly_pmt_y = monthly_payment_repayment(bal, rate, max(1, p.term_years - (y - 1)))
                bal, interest_paid, principal_paid = amortization_year(bal, rate, monthly_pmt_y, 12)
                mortgage_paid = (interest_paid + principal_paid)
            else:
                # Interest-only
                interest_paid = bal * (rate / 100.0)
                principal_paid = 0.0
                mortgage_paid = interest_paid

        # Ltd-owned taxable profit: rental net BEFORE principal; interest is deductible.
        # taxable = (gross - allowable costs - interest)
        taxable_profit = rent_net_before_finance - interest_paid
        if taxable_profit < 0:
            # realism: losses can be carried forward; we approximate by allowing negative (no tax now)
            # and carry forward against future profits.
            # We'll track a loss pool.
            pass

        # Loss carryforward
        if y == 1:
            loss_pool = 0.0  # initialize

        taxable_after_losses = taxable_profit
        if taxable_after_losses < 0:
            loss_pool += -taxable_after_losses
            taxable_after_losses = 0.0
        else:
            offset = min(loss_pool, taxable_after_losses)
            taxable_after_losses -= offset
            loss_pool -= offset

        corp_tax = corp_tax_on_profit(taxable_after_losses)

        # Post-tax cashflow (actual cash)
        post_tax_cf = rent_net_before_finance - mortgage_paid - corp_tax

        # Add ATED if toggled
        if p.apply_ated and p.ated_annual_charge > 0:
            post_tax_cf -= p.ated_annual_charge

        # Discount to real at t0 (end-of-year)
        real_cf = post_tax_cf / (1 + p.inflation_pct / 100.0) ** y

        cashflows_nominal.append(post_tax_cf)
        cashflows_real.append(real_cf)

        equity = prop_val - bal

        results.append(YearResult(
            year=y,
            rate_pct=rate,
            rent_gross=gross_rent,
            rent_net_before_finance=rent_net_before_finance,
            interest_paid=interest_paid,
            principal_paid=principal_paid,
            mortgage_paid=mortgage_paid,
            corp_tax=corp_tax,
            post_tax_cashflow=post_tax_cf,
            balance_end=bal,
            property_value_end=prop_val,
            equity_end=equity
        ))

        # Roll forward: rent and costs grow
        rent_m *= (1 + p.rent_growth_pct / 100.0)
        infl = (1 + p.inflation_pct / 100.0)
        opex_m *= infl
        insurance_m *= infl
        council_m *= infl
        service_m *= infl

    # Profit if sold each year
    sale_table = []
    cum_nom = 0.0
    cum_real = 0.0
    # cumulative includes time0 outflow
    cum_nom = cashflows_nominal[0]
    cum_real = cashflows_real[0]

    for yr in range(1, p.max_years + 1):
        r = results[yr - 1]
        cum_nom += cashflows_nominal[yr]
        cum_real += cashflows_real[yr]

        sell_costs = r.property_value_end * p.sell_cost_pct + p.sell_cost_fixed
        sale_net_before_tax = r.property_value_end - sell_costs - r.balance_end

        # Company “chargeable gain” taxed under corp tax (no annual CGT allowance for companies is a common point;
        # exact treatment depends on many details). GOV.UK: company pays corp tax on chargeable gain.
        gain_tax = 0.0
        if p.include_gain_in_corp_tax:
            # Simplified gain: (sale price - sell costs) - (purchase price + buy costs + SDLT)
            gain = (r.property_value_end - sell_costs) - (p.price + buy_costs + sdlt)
            gain = max(0.0, gain)
            gain_tax = corp_tax_on_profit(gain)  # treated as if stand-alone; conservative-ish
        sale_net = sale_net_before_tax - gain_tax

        profit_if_sold_nom = cum_nom + sale_net
        profit_if_sold_real = cum_real + (sale_net / (1 + p.inflation_pct / 100.0) ** yr)

        note = []
        if yr == p.sale_year: note.append("SALE_YEAR")
        if yr == p.term_years: note.append("FULL_OWNERSHIP")
        sale_table.append({
            "year": yr,
            "profit_if_sold_nominal": profit_if_sold_nom,
            "profit_if_sold_real": profit_if_sold_real,
            "sale_net_nominal": sale_net,
            "balance_end": r.balance_end,
            "equity_end": r.equity_end,
            "note": ",".join(note)
        })

    # Metrics
    def irr(cfs: List[float]) -> Optional[float]:
        # basic IRR via Newton; returns annual IRR approximated from yearly cashflows
        # cashflows are annual except index 0
        guess = 0.10
        for _ in range(60):
            npv = 0.0
            d = 0.0
            for t, cf in enumerate(cfs):
                npv += cf / ((1 + guess) ** t)
                if t > 0:
                    d -= t * cf / ((1 + guess) ** (t + 1))
            if abs(npv) < 1e-6:
                return guess
            if abs(d) < 1e-12:
                return None
            guess -= npv / d
            if guess <= -0.95:
                return None
        return None

    # Compute “sale at sale_year” IRR on nominal cashflows:
    # Build cashflow stream with sale proceeds inserted at sale_year and truncated
    sale_idx = p.sale_year
    if sale_idx > p.max_years:
        sale_idx = p.max_years

    sale_row = sale_table[sale_idx - 1]
    cfs_sale_nom = cashflows_nominal[:sale_idx + 1]
    cfs_sale_nom[-1] += sale_row["sale_net_nominal"]
    irr_nom = irr(cfs_sale_nom)

    # Debt Service Coverage Ratio (DSCR): net operating income / debt service.
    # Use sale_year snapshot + worst-year.
    dscrs = []
    for r in results[:p.sale_year]:
        debt = r.mortgage_paid if r.mortgage_paid > 0 else 0.0
        noi = r.rent_net_before_finance  # NOI before finance
        dscrs.append(noi / debt if debt > 0 else float("inf"))

    out = {
        "params": asdict(p),
        "initial_cash_out": initial_cash_out,
        "timeline": sale_table,
        "yearly": [asdict(x) for x in results],
        "irr_nominal_sale_year": irr_nom,
        "worst_dscr_to_sale": min(dscrs) if dscrs else None,
    }
    return out


# ----------------------------
# Monte Carlo runner
# ----------------------------

@dataclass
class MCSpec:
    n: int = 2000
    seed: int = 42

    # 95% ranges (explicitly shown as (lo, hi))
    house_growth_95: Tuple[float, float] = (-2.0, 6.0)
    rent_growth_95: Tuple[float, float] = (0.0, 6.0)
    inflation_95: Tuple[float, float] = (1.0, 8.0)
    rate_95: Tuple[float, float] = (2.0, 8.0)

    void_95: Tuple[float, float] = (0.0, 0.12)
    mgmt_fee_95: Tuple[float, float] = (0.08, 0.15)
    capex_95: Tuple[float, float] = (0.02, 0.12)

    maint_prob_95: Tuple[float, float] = (0.05, 0.25)
    maint_size_months_95: Tuple[float, float] = (0.5, 3.0)

def run_monte_carlo(base: Params, spec: MCSpec) -> Dict[str, object]:
    rng = random.Random(spec.seed)
    results = []

    for _ in range(spec.n):
        p = Params(**asdict(base))

        # Sample uncertain parameters (medians come from base)
        p.house_price_growth_pct = triangular_from_95(base.house_price_growth_pct, spec.house_growth_95[0], spec.house_growth_95[1], rng)
        p.rent_growth_pct = triangular_from_95(base.rent_growth_pct, spec.rent_growth_95[0], spec.rent_growth_95[1], rng)
        p.inflation_pct = triangular_from_95(base.inflation_pct, spec.inflation_95[0], spec.inflation_95[1], rng)
        p.annual_rate_pct = triangular_from_95(base.annual_rate_pct, spec.rate_95[0], spec.rate_95[1], rng)

        p.void_bad_debt_frac = triangular_from_95(base.void_bad_debt_frac, spec.void_95[0], spec.void_95[1], rng)
        p.letting_mgmt_fee_frac = triangular_from_95(base.letting_mgmt_fee_frac, spec.mgmt_fee_95[0], spec.mgmt_fee_95[1], rng)
        p.capex_reserve_frac = triangular_from_95(base.capex_reserve_frac, spec.capex_95[0], spec.capex_95[1], rng)

        p.maint_shock_prob_per_year = triangular_from_95(base.maint_shock_prob_per_year, spec.maint_prob_95[0], spec.maint_prob_95[1], rng)
        p.maint_shock_size_months_rent = triangular_from_95(base.maint_shock_size_months_rent, spec.maint_size_months_95[0], spec.maint_size_months_95[1], rng)

        out = simulate(p, rng=rng)
        # Take profit if sold at sale_year (nominal + real)
        row = out["timeline"][p.sale_year - 1]
        results.append({
            "profit_nom": row["profit_if_sold_nominal"],
            "profit_real": row["profit_if_sold_real"],
            "irr_nom": out["irr_nominal_sale_year"],
            "worst_dscr": out["worst_dscr_to_sale"],
        })

    # Summaries
    def pct(xs, q):
        xs2 = sorted(x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))
        if not xs2:
            return None
        i = int(round((len(xs2) - 1) * q))
        return xs2[i]

    profits_nom = [r["profit_nom"] for r in results]
    profits_real = [r["profit_real"] for r in results]
    irrs = [r["irr_nom"] for r in results if r["irr_nom"] is not None]
    dscrs = [r["worst_dscr"] for r in results if r["worst_dscr"] is not None]

    summary = {
        "sale_year": base.sale_year,
        "profit_nom_p05_p50_p95": (pct(profits_nom, 0.05), pct(profits_nom, 0.50), pct(profits_nom, 0.95)),
        "profit_real_p05_p50_p95": (pct(profits_real, 0.05), pct(profits_real, 0.50), pct(profits_real, 0.95)),
        "irr_nom_p05_p50_p95": (pct(irrs, 0.05), pct(irrs, 0.50), pct(irrs, 0.95)),
        "worst_dscr_p05_p50_p95": (pct(dscrs, 0.05), pct(dscrs, 0.50), pct(dscrs, 0.95)),
        "n": spec.n,
    }

    return {"summary": summary, "samples": results}


# ----------------------------
# Example usage
# ----------------------------

if __name__ == "__main__":
    base = Params(
        price=300_000.0,
        deposit_pct=0.25,
        term_years=25,
        mortgage_type="repayment",
        rent_monthly_start=1500.0,
        rent_growth_pct=3.0,              # +/- (0–6)% 95% in MC
        inflation_pct=3.5,                # +/- (1–8)% 95% in MC
        house_price_growth_pct=3.0,        # +/- (-2–6)% 95% in MC
        annual_rate_pct=5.0,               # +/- (2–8)% 95% in MC
        letting_mgmt_fee_frac=0.12,        # +/- (0.08–0.15)
        void_bad_debt_frac=0.08,           # +/- (0–0.12)
        capex_reserve_frac=0.06,           # +/- (0.02–0.12)
        sale_year=10,
        max_years=25,
        sdlt_mode="higher_additional",     # or "corp_17pct_worstcase" if you want worst-case enveloped risk
        sdlt_surcharge_pp=0.05,
        apply_ated=False,
        include_gain_in_corp_tax=True,
    )

    # Deterministic run
    out = simulate(base, rng=None)

    # Print timeline: profit if sold each year
    print("=== Timeline (profit if sold) ===")
    for row in out["timeline"]:
        y = row["year"]
        note = row["note"]
        print(f"Y{y:02d}: Nom £{row['profit_if_sold_nominal']:,.0f} | Real £{row['profit_if_sold_real']:,.0f} {('['+note+']') if note else ''}")

    print("\nIRR (nominal) at sale_year:", out["irr_nominal_sale_year"])
    print("Worst DSCR to sale_year:", out["worst_dscr_to_sale"])
    print("Initial cash out:", out["initial_cash_out"])

    # Monte Carlo run (optional)
    spec = MCSpec(n=2000, seed=42)
    mc = run_monte_carlo(base, spec)
    print("\n=== Monte Carlo summary @ sale_year ===")
    print(mc["summary"])

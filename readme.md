# UK Buy-to-Let Investment Model (Ltd-Owned)

A realism-first, interactive financial analysis tool for evaluating UK
buy-to-let property investments held via a **limited company**.

This tool is designed to support **large financial decisions** by making
assumptions explicit, modelling risk, and separating **company profit**
from **personal cash**.

---

## What This App Does

This application simulates the full lifecycle of a UK residential
buy-to-let property owned by a **Ltd company**, and evaluates:

- Annual rental cashflows
- Mortgage repayments and equity build-up
- Corporation tax and chargeable gains
- Optional dividend extraction to personal cash
- Profit if sold in **each year**
- Nominal vs real (inflation-adjusted) outcomes
- Risk and uncertainty via Monte Carlo simulation

Results are presented via:
- Interactive tables (Ltd vs Personal)
- Pop-out graphs with uncertainty bands
- Clear summary metrics (IRR, cash out, DSCR)

---

## Key Features

### Property & Mortgage
- Repayment or interest-only mortgages
- Optional balloon repayment at sale
- Explicit per-year interest rate paths
- Equity tracking over time

### Rental Income & Costs
- Rent growth
- Operating costs with inflation
- Voids and bad debt
- Letting / management fees
- Capex reserve
- Stochastic maintenance shocks (non-linear realism)

### UK Tax Modelling
- Corporation tax on operating profit
- Chargeable gains on sale
- SDLT (standard / higher-rate / corporate worst-case)
- Optional ATED annual charge

### Personal Cash Extraction
- Optional dividend extraction layer
- Annual or sale-only extraction modes
- Explicit dividend tax rate and allowance
- Outputs represent **money that actually reaches your bank**

### Risk & Uncertainty
- Monte Carlo simulation
- User-defined 95% ranges for key variables
- Median and 5–95% outcome bands
- Deterministic + stochastic comparison

---

## Explicit Assumptions (Important)

This tool is intentionally transparent.  
If an assumption is not stated here, it should not be trusted.

---

### Ownership Structure
- Property is owned by a **UK limited company**
- No joint ownership or partnerships
- Single asset per simulation

---

### Tax Assumptions

#### Corporation Tax
- Flat marginal corporation tax approximation
- No group relief
- No loss carry-back / carry-forward optimisation

#### Dividend Tax
- Dividend extraction is optional
- When enabled:
  - A **flat marginal dividend tax rate** is applied
  - Dividend allowance is applied once per year
- **No PAYE salary interaction**
- **No NICs** (dividends only)
- Does **not** model crossing tax bands mid-year

This is a **first-order wealth translation**, not a full personal tax planner.

---

### Mortgage & Financing
- Mortgage terms are fixed for the full term
- No automatic refinancing
- Interest-only balloon (if enabled) is assumed payable at sale
- No early repayment charges

---

### Inflation
- Inflation is applied consistently to:
  - Operating costs
  - Rents (if specified)
- Real values are discounted using the stated inflation rate

---

### Monte Carlo Simulation
- Uses triangular distributions
- 95% ranges are **user-defined**
- Assumes independence between variables
- No regime-switching or correlated macro shocks

Monte Carlo is used to explore **uncertainty**, not predict outcomes.

---

## What This Tool Is NOT

- ❌ Tax advice
- ❌ A replacement for an accountant
- ❌ A guarantee of returns
- ❌ A “make deals look good” spreadsheet

It is a **decision support tool**.

---

## How to Run (Python)

```bash
pip install -r requirements.txt
python app.py

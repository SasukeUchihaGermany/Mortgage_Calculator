"""
Interactive Tkinter application for the UK buy‑to‑let investment model.

This GUI wraps the core simulation functions defined in ``model.py`` and
provides a form for adjusting all model parameters. The user can
enter values for each field (with sensible defaults pre‑populated),
optionally specify a custom per‑year interest rate path, and choose
whether or not to apply a simple dividend extraction layer. When the
"Run Simulation" button is pressed the underlying model is executed
using the supplied parameters and the results are displayed in the
interface. A profit‑timeline chart shows the profit if the property
were sold at the end of each year along with 5th–95th percentile
uncertainty bands derived from a Monte Carlo run. If the dividend
layer is enabled then a second line showing the personal after‑tax
profit trajectory is also plotted.

Sources used to inform tax logic and mortgage assumptions:

* GOV.UK explains that the dividend allowance is £500 for the
  2024–25 tax year and lists the dividend tax rates for different
  income bands【726006200913765†L116-L133】. A press release from Deloitte’s
  Autumn Budget 2025 coverage confirms that, from April 2026, the
  dividend ordinary and upper tax rates will rise to 10.75% and
  35.75% respectively, while the additional rate remains at 39.35%
  【612842057230984†L70-L75】. These figures are used as default dividend
  tax rates in the personal cash‑flow calculations.

* Guidance from a UK conveyancing firm notes that when a fixed‑rate
  mortgage expires the loan automatically reverts to the lender’s
  standard variable rate (SVR), which usually carries a higher
  interest rate【788349540079157†L192-L196】. This behaviour is modelled via
  an optional per‑year interest rate path: if the user enters a
  comma‑separated sequence of rates the simulation will apply the
  specified rate in each corresponding year and revert to the base
  rate thereafter.

The GUI depends on Tkinter (for widgets) and matplotlib (for the
chart). Ensure these packages are available in your Python environment.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Tuple, Optional
import random

import matplotlib
matplotlib.use("Agg")  # use a non‑interactive backend for safety
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import model  # the core simulation module (copy of pasted.txt)

from dataclasses import asdict

# -----------------------------------------------------------------------------
# Colour theme definitions for a dark, blue/grey interface.  These constants
# control the look and feel of the Tkinter UI.  Using named colours rather than
# magic strings makes it straightforward to tweak the palette later.
BG = "#0f172a"         # primary background (very dark blue/grey)
PANEL = "#111827"      # panels and entry backgrounds
FG = "#e5e7eb"         # primary foreground (light grey)
ACCENT = "#3b82f6"     # accent colour (medium blue)
ACCENT_ACTIVE = "#2563eb"  # accent when buttons are active/hovered
GRID = "#1f2937"       # gridlines for charts and table separators

def pretty_label(name: str) -> str:
    """
    Convert an internal parameter name into a more human‑readable label.

    Replaces underscores with spaces and the suffix ``_pct`` with a literal
    ``%`` sign, while title‑casing the result.  For example, ``rent_growth_pct``
    becomes ``Rent Growth %``.  This helper is used throughout the GUI to
    automatically transform dataclass field names into UI labels.
    """
    label = name.replace("_pct", " (%)").replace("_frac", " (fraction)")
    label = label.replace("_", " ")
    return label

def open_table_window(title: str, rows: List[Dict[str, object]], inputs: Dict[str, object], outputs: Dict[str, object]) -> None:
    """
    Present a new window with a tabular view of per‑year results and a summary
    of inputs and outputs.

    Parameters
    ----------
    title: str
        Title of the window (e.g. "Ltd – Profit if Sold").
    rows: List[Dict[str, object]]
        Each dict should contain at least ``year``, ``nominal`` and ``real`` keys.
    inputs: Dict[str, object]
        Mapping of input parameter names to their numeric/string values used in
        the simulation.  Displayed in the lower section of the window.
    outputs: Dict[str, object]
        Mapping of summary output names (e.g. IRR) to values.  Displayed after
        the inputs.
    """
    win = tk.Toplevel()
    win.title(title)
    win.configure(bg=BG)

    # Treeview for the table of year‑by‑year profits
    tree = ttk.Treeview(win, columns=("year", "nom", "real"), show="headings")
    tree.heading("year", text="Year")
    tree.heading("nom", text="Profit (£ nominal)")
    tree.heading("real", text="Profit (£ real)")
    tree.column("year", width=60, anchor="center")
    tree.column("nom", width=180, anchor="e")
    tree.column("real", width=180, anchor="e")
    tree.pack(fill="both", expand=True, padx=10, pady=10)

    # Populate the table
    for row in rows:
        year = row.get("year")
        nom = row.get("nominal", row.get("profit_if_sold_nominal"))
        real = row.get("real", row.get("profit_if_sold_real"))
        # Format numbers nicely
        nom_str = f"{nom:,.0f}" if isinstance(nom, (int, float)) else str(nom)
        real_str = f"{real:,.0f}" if isinstance(real, (int, float)) else str(real)
        tree.insert("", "end", values=(year, nom_str, real_str))

    # Text area for inputs and outputs summary
    meta = tk.Text(win, bg=PANEL, fg=FG, insertbackground=FG, font=("JetBrains Mono", 10), height=14)
    meta.pack(fill="x", padx=10, pady=(0, 10))
    meta.insert("end", "INPUTS\n")
    for k, v in inputs.items():
        meta.insert("end", f"  {pretty_label(k)}: {v}\n")
    meta.insert("end", "\nOUTPUTS\n")
    for k, v in outputs.items():
        meta.insert("end", f"  {k}: {v}\n")
    meta.config(state="disabled")


def irr(values: List[float]) -> Optional[float]:
    """Compute an internal rate of return given a list of cash flows.

    The implementation is adapted from the irr function defined
    inside model.simulate. It uses a simple Newton iteration to
    approximate the discount rate that drives the net present value
    of the cash flow stream to zero. If the iteration fails or the
    derivative is too small the function returns None.
    """
    guess = 0.10
    for _ in range(60):
        npv = 0.0
        d = 0.0
        for t, cf in enumerate(values):
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


class InvestmentApp:
    """Main application class for the interactive investment model."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("UK Buy‑to‑Let Investment Model")

        # Set a reasonable default size so that all input groups are visible without scrolling.
        # This can be adjusted later by the user via window resizing.
        self.root.geometry("1200x900")

        # Apply global dark theme colours
        self.root.configure(bg=BG)
        style = ttk.Style(self.root)
        # Use the default theme and override key elements
        try:
            style.theme_use("default")
        except Exception:
            pass
        style.configure(
            ".", background=BG, foreground=FG, fieldbackground=PANEL, bordercolor=GRID
        )
        # Buttons with accent colour
        style.configure("TButton", background=ACCENT, foreground="white", padding=6)
        style.map("TButton", background=[("active", ACCENT_ACTIVE)])
        # Entry/combobox/checkbox backgrounds
        style.configure("TEntry", fieldbackground=PANEL, foreground=FG)
        style.configure("TCombobox", fieldbackground=PANEL, foreground=FG)
        style.configure("TCheckbutton", background=BG, foreground=FG)
        style.configure("TLabel", background=BG, foreground=FG)
        # Treeview styling: headings and rows contrast with panel background
        style.configure("Treeview", background=PANEL, foreground=FG, fieldbackground=PANEL, bordercolor=GRID)
        style.configure("Treeview.Heading", background=PANEL, foreground=FG)
        # LabelFrame styling: ensure section headers match dark theme
        style.configure("TLabelframe", background=BG, foreground=FG, bordercolor=GRID)
        style.configure("TLabelframe.Label", background=BG, foreground=FG)

        # Prepare containers for parameter variables and extra settings
        # self.vars holds model.Params fields; self.mc_range_vars holds uncertainty ranges;
        # custom vars (MC settings, personal settings) are stored as attributes.
        self.vars: Dict[str, tk.Variable] = {}
        self.mc_range_vars: Dict[str, tk.Variable] = {}

        # Build the sectioned UI for inputs
        self._build_sections()

        # Button to run simulation
        self.run_button = ttk.Button(
            self.root, text="Run Simulation", command=self.run_simulation
        )
        self.run_button.pack(pady=5)

        # Output text area: enlarge font and apply theme colours.  Use fill=X so the chart below gets
        # most of the vertical space.
        self.output_text = tk.Text(
            self.root,
            height=8,
            wrap="word",
            bg=PANEL,
            fg=FG,
            insertbackground=FG,
            font=("JetBrains Mono", 20),
        )
        self.output_text.pack(fill=tk.X, expand=False, padx=10, pady=5)

        # Figure for chart.  The canvas will be created lazily in a pop‑out window on demand.
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        # Apply dark theme to the Matplotlib figure
        self.figure.patch.set_facecolor(BG)
        self.ax.set_facecolor(PANEL)
        # Canvas and graph window will be created when running the simulation
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.graph_window: Optional[tk.Toplevel] = None

    def show_graph(self) -> None:
        """
        Create or update the pop‑out window that displays the profit trajectory chart.
        This method constructs a Toplevel window (if one does not already exist) and
        embeds the current matplotlib figure into it.  The canvas is redrawn on
        each call.
        """
        # If no graph window exists or it has been closed, create a new one
        if self.graph_window is None or not self.graph_window.winfo_exists():
            self.graph_window = tk.Toplevel(self.root)
            self.graph_window.title("Profit trajectory with uncertainty bands")
            self.graph_window.configure(bg=BG)
            # Create a fresh canvas attached to the figure
            self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_window)
            canvas_widget = self.canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True)
        # Redraw the figure on the canvas
        if self.canvas is not None:
            self.canvas.draw()

    def _create_param_inputs(self) -> None:
        """Create labelled input controls for each model parameter."""
        p = model.Params()
        row = 0

        # Helper to add a label and entry/checkbox/dropdown
        def add_entry(name: str, default_value, var_type: type, tooltip: str = "", choices: List[str] = None):
            nonlocal row
            # Human‑friendly label using pretty_label helper
            label = ttk.Label(self.param_inner, text=pretty_label(name))
            label.grid(row=row, column=0, sticky=tk.W, padx=2, pady=2)
            if choices:
                var = tk.StringVar(value=str(default_value))
                entry = ttk.Combobox(
                    self.param_inner,
                    textvariable=var,
                    values=choices,
                    state="readonly",
                )
                entry.grid(row=row, column=1, sticky=tk.W, padx=2, pady=2)
            elif var_type is bool:
                var = tk.BooleanVar(value=bool(default_value))
                entry = ttk.Checkbutton(self.param_inner, variable=var)
                entry.grid(row=row, column=1, sticky=tk.W, padx=2, pady=2)
            else:
                var = tk.StringVar(value=str(default_value))
                entry = ttk.Entry(self.param_inner, textvariable=var, width=12)
                entry.grid(row=row, column=1, sticky=tk.W, padx=2, pady=2)
            self.vars[name] = var
            row += 1

        # Populate entries for each Params field.
        # Purchase
        add_entry("price", p.price, float)
        add_entry("deposit_pct", p.deposit_pct, float)

        # Mortgage
        add_entry("term_years", p.term_years, int)
        add_entry("mortgage_type", p.mortgage_type, str, choices=["repayment", "interest_only"])
        add_entry("interest_only_balloon", p.interest_only_balloon, bool)

        # Income
        add_entry("rent_monthly_start", p.rent_monthly_start, float)
        add_entry("rent_growth_pct", p.rent_growth_pct, float)

        # Operating costs
        add_entry("opex_monthly_start", p.opex_monthly_start, float)
        add_entry("council_tax_landlord_monthly", p.council_tax_landlord_monthly, float)
        add_entry("insurance_monthly", p.insurance_monthly, float)
        add_entry("ground_rent_service_charge_monthly", p.ground_rent_service_charge_monthly, float)

        # Variable cost fractions
        add_entry("void_bad_debt_frac", p.void_bad_debt_frac, float)
        add_entry("letting_mgmt_fee_frac", p.letting_mgmt_fee_frac, float)
        add_entry("capex_reserve_frac", p.capex_reserve_frac, float)

        # Maintenance shocks
        add_entry("maint_shock_prob_per_year", p.maint_shock_prob_per_year, float)
        add_entry("maint_shock_size_months_rent", p.maint_shock_size_months_rent, float)

        # Transactions
        add_entry("buy_cost_pct", p.buy_cost_pct, float)
        add_entry("buy_cost_fixed", p.buy_cost_fixed, float)
        add_entry("sell_cost_pct", p.sell_cost_pct, float)
        add_entry("sell_cost_fixed", p.sell_cost_fixed, float)

        # SDLT options
        add_entry("apply_sdlt", p.apply_sdlt, bool)
        add_entry("sdlt_mode", p.sdlt_mode, str, choices=["standard", "higher_additional", "corp_17pct_worstcase"])
        add_entry("sdlt_surcharge_pp", p.sdlt_surcharge_pp, float)

        # ATED
        add_entry("apply_ated", p.apply_ated, bool)
        add_entry("ated_annual_charge", p.ated_annual_charge, float)

        # Price/inflation paths
        add_entry("inflation_pct", p.inflation_pct, float)
        add_entry("house_price_growth_pct", p.house_price_growth_pct, float)

        # Interest rate path behaviour
        add_entry("annual_rate_pct", p.annual_rate_pct, float)
        # Provide a custom interest rate path input (comma separated)
        add_entry("interest_rate_path", "", str)

        # Horizon
        add_entry("max_years", p.max_years, int)
        add_entry("sale_year", p.sale_year, int)

        # Company level
        add_entry("include_gain_in_corp_tax", p.include_gain_in_corp_tax, bool)

        # Monte Carlo spec
        self.mc_n_var = tk.StringVar(value="200")
        self.mc_seed_var = tk.StringVar(value="42")
        ttk.Label(self.param_inner, text="MC samples").grid(row=row, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Entry(self.param_inner, textvariable=self.mc_n_var, width=12).grid(row=row, column=1, sticky=tk.W, padx=2, pady=2)
        row += 1
        ttk.Label(self.param_inner, text="MC seed").grid(row=row, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Entry(self.param_inner, textvariable=self.mc_seed_var, width=12).grid(row=row, column=1, sticky=tk.W, padx=2, pady=2)
        row += 1

        # Personal cash‑flow settings
        self.personal_enabled = tk.BooleanVar(value=True)
        ttk.Label(self.param_inner, text="Apply dividend tax").grid(row=row, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Checkbutton(self.param_inner, variable=self.personal_enabled).grid(row=row, column=1, sticky=tk.W, padx=2, pady=2)
        row += 1

    def _build_sections(self) -> None:
        """
        Build the input form with logical sections laid out side by side.

        This replaces the old scrollable parameter canvas with a set of labelled
        frames arranged in two columns. Each group of related inputs is placed
        in its own `ttk.LabelFrame` for clarity.  Model parameters are stored
        in `self.vars`, Monte Carlo uncertainty ranges in `self.mc_range_vars`,
        and additional personal/MC settings on the instance.
        """
        # Defaults from model parameters and MC spec
        p = model.Params()
        spec_default = model.MCSpec()

        # Create a parent frame for all input sections
        form = tk.Frame(self.root, bg=BG)
        form.pack(fill=tk.X, padx=10, pady=10)

        # Two columns for better use of horizontal space
        left_col = tk.Frame(form, bg=BG)
        right_col = tk.Frame(form, bg=BG)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # Helper functions
        def ensure_row_counter(frame):
            """
            Ensure that the given frame has a '_next_row' attribute for grid row placement.
            Initialize if not present.
            """
            if not hasattr(frame, "_next_row"):
                frame._next_row = 0

        def add_param_field(frame, key: str, default_value, var_type: type, choices: List[str] = None):
            """Add a single input for a Params field and register its variable using grid layout."""
            ensure_row_counter(frame)
            row = frame._next_row
            # Determine variable type
            if choices:
                var = tk.StringVar(value=str(default_value))
                widget = ttk.Combobox(frame, textvariable=var, values=choices, state="readonly", width=12)
            elif var_type is bool:
                var = tk.BooleanVar(value=bool(default_value))
                widget = ttk.Checkbutton(frame, variable=var)
            else:
                var = tk.StringVar(value=str(default_value))
                widget = ttk.Entry(frame, textvariable=var, width=12)
            # Create label and place using grid
            lbl = ttk.Label(frame, text=pretty_label(key))
            lbl.grid(row=row, column=0, sticky="w", padx=2, pady=2)
            widget.grid(row=row, column=1, sticky="e", padx=2, pady=2)
            # Register variable
            self.vars[key] = var
            # Increment row counter
            frame._next_row += 1

        def add_custom_bool(frame, label_text: str, var: tk.BooleanVar):
            ensure_row_counter(frame)
            row = frame._next_row
            lbl = ttk.Label(frame, text=label_text)
            lbl.grid(row=row, column=0, sticky="w", padx=2, pady=2)
            cb = ttk.Checkbutton(frame, variable=var)
            cb.grid(row=row, column=1, sticky="e", padx=2, pady=2)
            frame._next_row += 1

        def add_custom_entry(frame, label_text: str, var: tk.StringVar):
            ensure_row_counter(frame)
            row = frame._next_row
            lbl = ttk.Label(frame, text=label_text)
            lbl.grid(row=row, column=0, sticky="w", padx=2, pady=2)
            ent = ttk.Entry(frame, textvariable=var, width=12)
            ent.grid(row=row, column=1, sticky="e", padx=2, pady=2)
            frame._next_row += 1

        def add_custom_choice(frame, label_text: str, var: tk.StringVar, choices: List[str]):
            ensure_row_counter(frame)
            row = frame._next_row
            lbl = ttk.Label(frame, text=label_text)
            lbl.grid(row=row, column=0, sticky="w", padx=2, pady=2)
            cmb = ttk.Combobox(frame, textvariable=var, values=choices, state="readonly", width=12)
            cmb.grid(row=row, column=1, sticky="e", padx=2, pady=2)
            frame._next_row += 1

        def add_range_field(frame, label_text: str, lo_default: float, hi_default: float, key_lo: str, key_hi: str):
            """Add two entries for low/high values of an uncertainty range using grid layout."""
            ensure_row_counter(frame)
            row = frame._next_row
            lbl = ttk.Label(frame, text=label_text)
            lbl.grid(row=row, column=0, sticky="w", padx=2, pady=2)
            var_lo = tk.StringVar(value=str(lo_default))
            var_hi = tk.StringVar(value=str(hi_default))
            ent_lo = ttk.Entry(frame, textvariable=var_lo, width=6)
            ent_hi = ttk.Entry(frame, textvariable=var_hi, width=6)
            ent_lo.grid(row=row, column=1, sticky="e", padx=2, pady=2)
            ent_hi.grid(row=row, column=2, sticky="e", padx=2, pady=2)
            self.mc_range_vars[key_lo] = var_lo
            self.mc_range_vars[key_hi] = var_hi
            frame._next_row += 1

        # -----------------------------------------------------------------
        # Left column groups: Purchase & Mortgage, Income & Growth, Costs & Fees
        # -----------------------------------------------------------------
        purchase_frame = ttk.LabelFrame(left_col, text="Purchase & Mortgage", padding=(5, 5))
        purchase_frame.pack(fill=tk.X, padx=5, pady=5)
        # Configure columns so labels and inputs align properly
        purchase_frame.columnconfigure(0, weight=1)
        purchase_frame.columnconfigure(1, weight=1)
        purchase_frame.columnconfigure(2, weight=1)
        # Purchase & mortgage fields
        add_param_field(purchase_frame, "price", p.price, float)
        add_param_field(purchase_frame, "deposit_pct", p.deposit_pct, float)
        add_param_field(purchase_frame, "term_years", p.term_years, int)
        add_param_field(purchase_frame, "mortgage_type", p.mortgage_type, str, choices=["repayment", "interest_only"])
        add_param_field(purchase_frame, "interest_only_balloon", p.interest_only_balloon, bool)
        add_param_field(purchase_frame, "annual_rate_pct", p.annual_rate_pct, float)
        add_param_field(purchase_frame, "interest_rate_path", "", str)
        # Horizon / sale fields
        horizon_frame = ttk.LabelFrame(left_col, text="Horizon & Sale", padding=(5, 5))
        horizon_frame.pack(fill=tk.X, padx=5, pady=5)
        horizon_frame.columnconfigure(0, weight=1)
        horizon_frame.columnconfigure(1, weight=1)
        horizon_frame.columnconfigure(2, weight=1)
        add_param_field(horizon_frame, "max_years", p.max_years, int)
        add_param_field(horizon_frame, "sale_year", p.sale_year, int)

        # Income & Growth fields
        income_frame = ttk.LabelFrame(left_col, text="Income & Growth", padding=(5, 5))
        income_frame.pack(fill=tk.X, padx=5, pady=5)
        income_frame.columnconfigure(0, weight=1)
        income_frame.columnconfigure(1, weight=1)
        income_frame.columnconfigure(2, weight=1)
        add_param_field(income_frame, "rent_monthly_start", p.rent_monthly_start, float)
        add_param_field(income_frame, "rent_growth_pct", p.rent_growth_pct, float)
        add_param_field(income_frame, "inflation_pct", p.inflation_pct, float)
        add_param_field(income_frame, "house_price_growth_pct", p.house_price_growth_pct, float)

        # Costs & Fees
        costs_frame = ttk.LabelFrame(left_col, text="Costs & Fees", padding=(5, 5))
        costs_frame.pack(fill=tk.X, padx=5, pady=5)
        costs_frame.columnconfigure(0, weight=1)
        costs_frame.columnconfigure(1, weight=1)
        costs_frame.columnconfigure(2, weight=1)
        add_param_field(costs_frame, "opex_monthly_start", p.opex_monthly_start, float)
        add_param_field(costs_frame, "council_tax_landlord_monthly", p.council_tax_landlord_monthly, float)
        add_param_field(costs_frame, "insurance_monthly", p.insurance_monthly, float)
        add_param_field(costs_frame, "ground_rent_service_charge_monthly", p.ground_rent_service_charge_monthly, float)
        add_param_field(costs_frame, "void_bad_debt_frac", p.void_bad_debt_frac, float)
        add_param_field(costs_frame, "letting_mgmt_fee_frac", p.letting_mgmt_fee_frac, float)
        add_param_field(costs_frame, "capex_reserve_frac", p.capex_reserve_frac, float)
        add_param_field(costs_frame, "maint_shock_prob_per_year", p.maint_shock_prob_per_year, float)
        add_param_field(costs_frame, "maint_shock_size_months_rent", p.maint_shock_size_months_rent, float)

        # -----------------------------------------------------------------
        # Right column groups: Transactions & Taxes, Personal Extraction, Monte Carlo, Uncertainty Ranges
        # -----------------------------------------------------------------
        transaction_frame = ttk.LabelFrame(right_col, text="Transactions & Taxes", padding=(5, 5))
        transaction_frame.pack(fill=tk.X, padx=5, pady=5)
        transaction_frame.columnconfigure(0, weight=1)
        transaction_frame.columnconfigure(1, weight=1)
        transaction_frame.columnconfigure(2, weight=1)
        add_param_field(transaction_frame, "buy_cost_pct", p.buy_cost_pct, float)
        add_param_field(transaction_frame, "buy_cost_fixed", p.buy_cost_fixed, float)
        add_param_field(transaction_frame, "sell_cost_pct", p.sell_cost_pct, float)
        add_param_field(transaction_frame, "sell_cost_fixed", p.sell_cost_fixed, float)
        add_param_field(transaction_frame, "apply_sdlt", p.apply_sdlt, bool)
        add_param_field(transaction_frame, "sdlt_mode", p.sdlt_mode, str, choices=["standard", "higher_additional", "corp_17pct_worstcase"])
        add_param_field(transaction_frame, "sdlt_surcharge_pp", p.sdlt_surcharge_pp, float)
        add_param_field(transaction_frame, "apply_ated", p.apply_ated, bool)
        add_param_field(transaction_frame, "ated_annual_charge", p.ated_annual_charge, float)
        add_param_field(transaction_frame, "include_gain_in_corp_tax", p.include_gain_in_corp_tax, bool)

        # Personal extraction settings
        personal_frame = ttk.LabelFrame(right_col, text="Personal Extraction", padding=(5, 5))
        personal_frame.pack(fill=tk.X, padx=5, pady=5)
        personal_frame.columnconfigure(0, weight=1)
        personal_frame.columnconfigure(1, weight=1)
        personal_frame.columnconfigure(2, weight=1)
        # Apply dividend tax
        self.personal_enabled = tk.BooleanVar(value=True)
        add_custom_bool(personal_frame, "Apply dividend tax", self.personal_enabled)
        # Extraction mode: annual or sale only
        self.dividend_mode_var = tk.StringVar(value="annual")
        add_custom_choice(personal_frame, "Extraction mode", self.dividend_mode_var, ["annual", "sale_only"])
        # Dividend tax rate and allowance
        self.dividend_tax_rate_var = tk.StringVar(value="0.1075")
        add_custom_entry(personal_frame, "Dividend tax rate", self.dividend_tax_rate_var)
        self.dividend_allowance_var = tk.StringVar(value="500")
        add_custom_entry(personal_frame, "Dividend allowance", self.dividend_allowance_var)

        # Monte Carlo core settings
        mc_frame = ttk.LabelFrame(right_col, text="Monte Carlo", padding=(5, 5))
        mc_frame.pack(fill=tk.X, padx=5, pady=5)
        mc_frame.columnconfigure(0, weight=1)
        mc_frame.columnconfigure(1, weight=1)
        mc_frame.columnconfigure(2, weight=1)
        self.mc_n_var = tk.StringVar(value="200")
        add_custom_entry(mc_frame, "MC samples", self.mc_n_var)
        self.mc_seed_var = tk.StringVar(value="42")
        add_custom_entry(mc_frame, "MC seed", self.mc_seed_var)

        # Uncertainty ranges (95%) for Monte Carlo
        mc_range_frame = ttk.LabelFrame(right_col, text="Uncertainty Ranges (95%)", padding=(5, 5))
        mc_range_frame.pack(fill=tk.X, padx=5, pady=5)
        mc_range_frame.columnconfigure(0, weight=1)
        mc_range_frame.columnconfigure(1, weight=1)
        mc_range_frame.columnconfigure(2, weight=1)
        # Each call defines lo and hi fields; use defaults from spec_default
        add_range_field(mc_range_frame, "House price growth (%)", spec_default.house_growth_95[0], spec_default.house_growth_95[1], "house_growth_lo", "house_growth_hi")
        add_range_field(mc_range_frame, "Rent growth (%)", spec_default.rent_growth_95[0], spec_default.rent_growth_95[1], "rent_growth_lo", "rent_growth_hi")
        add_range_field(mc_range_frame, "Inflation (%)", spec_default.inflation_95[0], spec_default.inflation_95[1], "inflation_lo", "inflation_hi")
        add_range_field(mc_range_frame, "Interest rate (%)", spec_default.rate_95[0], spec_default.rate_95[1], "rate_lo", "rate_hi")
        add_range_field(mc_range_frame, "Void/bad debt (fraction)", spec_default.void_95[0], spec_default.void_95[1], "void_lo", "void_hi")
        add_range_field(mc_range_frame, "Mgmt fee (fraction)", spec_default.mgmt_fee_95[0], spec_default.mgmt_fee_95[1], "mgmt_lo", "mgmt_hi")
        add_range_field(mc_range_frame, "Capex reserve (fraction)", spec_default.capex_95[0], spec_default.capex_95[1], "capex_lo", "capex_hi")
        add_range_field(mc_range_frame, "Maint shock prob", spec_default.maint_prob_95[0], spec_default.maint_prob_95[1], "maint_prob_lo", "maint_prob_hi")
        add_range_field(mc_range_frame, "Maint shock size (months)", spec_default.maint_size_months_95[0], spec_default.maint_size_months_95[1], "maint_size_lo", "maint_size_hi")

        # Remove any leftover references to the old scrollable parameter grid.  All
        # personal dividend settings are added above in the `personal_frame`.

    def _parse_params(self) -> Optional[model.Params]:
        """Convert UI values into a Params object. Returns None on error."""
        try:
            kwargs: Dict[str, object] = {}
            for name, var in self.vars.items():
                val_str = var.get()
                if name in {"mortgage_type", "sdlt_mode", "interest_rate_path"}:
                    kwargs[name] = val_str.strip() if val_str else (None if name == "interest_rate_path" else getattr(model.Params(), name))
                elif name in {"interest_only_balloon", "apply_sdlt", "apply_ated", "include_gain_in_corp_tax"}:
                    kwargs[name] = bool(var.get())
                elif name in {"term_years", "max_years", "sale_year"}:
                    kwargs[name] = int(val_str)
                else:
                    kwargs[name] = float(val_str)
            # Parse interest_rate_path
            ir_path_raw = kwargs.get("interest_rate_path")
            if ir_path_raw:
                parts = [s.strip() for s in ir_path_raw.split(',') if s.strip()]
                kwargs["interest_rate_path"] = [float(x) for x in parts]
            else:
                kwargs["interest_rate_path"] = None
            return model.Params(**kwargs)
        except Exception as e:
            messagebox.showerror("Invalid input", f"Error parsing inputs: {e}")
            return None

    def run_simulation(self) -> None:
        """Run the model with current parameters and update output and chart."""
        base = self._parse_params()
        if base is None:
            return
        # Ensure sale_year does not exceed max_years
        if base.sale_year > base.max_years:
            messagebox.showwarning(
                "Adjust sale year",
                f"Sale year {base.sale_year} exceeds max_years {base.max_years}. It has been capped.",
            )
            base.sale_year = base.max_years

        # Deterministic run
        out = model.simulate(base, rng=None)

        # Compute personal cash flows if enabled
        personal_profit_if_sold_nominal: List[float] = []
        personal_irr_nominal = None
        if self.personal_enabled.get():
            # Parse dividend settings
            try:
                rate = float(self.dividend_tax_rate_var.get())
                allowance = float(self.dividend_allowance_var.get())
            except ValueError:
                messagebox.showerror("Input error", "Dividend rate and allowance must be numeric.")
                return
            mode = self.dividend_mode_var.get() or "annual"
            # Annual extraction: apply dividend tax each year
            if mode == "annual":
                cashflows_nom = [-out["initial_cash_out"]]
                cashflows_nom_personal = [-out["initial_cash_out"]]
                # Build per-year cash flows
                for r in out["yearly"]:
                    cf = r["post_tax_cashflow"]
                    cashflows_nom.append(cf)
                    # apply dividend tax to each year's cash flow
                    taxable = max(0.0, cf - allowance)
                    personal_cf = cf - taxable * rate
                    cashflows_nom_personal.append(personal_cf)
                # Compute sale proceeds for nominal and personal at sale_year
                sale_year = base.sale_year
                sale_row = out["timeline"][sale_year - 1]
                sale_net_nom = sale_row["sale_net_nominal"]
                # Add sale proceeds to nominal cash flow
                cashflows_nom_with_sale = cashflows_nom[: sale_year + 1]
                cashflows_nom_with_sale[-1] += sale_net_nom
                # Apply dividend tax to sale proceeds
                personal_sale_net = sale_net_nom - max(0.0, sale_net_nom - allowance) * rate
                cashflows_personal_with_sale = cashflows_nom_personal[: sale_year + 1]
                cashflows_personal_with_sale[-1] += personal_sale_net
                # Personal IRR on annual extraction stream
                personal_irr_nominal = irr(cashflows_personal_with_sale)
                # Construct personal profit if sold each year
                cum_personal = cashflows_nom_personal[0]
                for yr_idx, row in enumerate(out["timeline"], start=1):
                    cum_personal += cashflows_nom_personal[yr_idx]
                    sale_net = row["sale_net_nominal"]
                    sale_net_personal = sale_net - max(0.0, sale_net - allowance) * rate
                    personal_profit = cum_personal + sale_net_personal
                    personal_profit_if_sold_nominal.append(personal_profit)
            # Sale-only extraction: accumulate profits and extract once at sale
            elif mode == "sale_only":
                cum_cash = -out["initial_cash_out"]
                for row in out["timeline"]:
                    # add annual post-tax cashflow to cum_cash (no dividend tax applied annually)
                    year = row["year"]
                    cf = out["yearly"][year - 1]["post_tax_cashflow"]
                    cum_cash += cf
                    sale_net = row["sale_net_nominal"]
                    # lumps sum equals cum_cash + sale proceeds
                    lumps = cum_cash + sale_net
                    # dividend tax applied once at sale
                    personal_net = lumps - max(0.0, lumps - allowance) * rate
                    personal_profit_if_sold_nominal.append(personal_net)
                # Personal IRR: negative initial cash out, zeros until sale_year, then lumps net
                sale_year = base.sale_year
                # lumps net at sale year (1-indexed)
                if 1 <= sale_year <= len(personal_profit_if_sold_nominal):
                    lumps_net = personal_profit_if_sold_nominal[sale_year - 1]
                    cashflows_personal = [-out["initial_cash_out"]] + [0.0] * (sale_year - 1) + [lumps_net]
                    personal_irr_nominal = irr(cashflows_personal)
            else:
                messagebox.showerror("Input error", f"Unknown extraction mode: {mode}")
                return
        # Monte Carlo run for profit uncertainty bands
        try:
            mc_n = int(self.mc_n_var.get())
            mc_seed = int(self.mc_seed_var.get())
        except ValueError:
            messagebox.showerror("Input error", "Monte Carlo sample count and seed must be integers.")
            return
        # Parse user-defined 95% ranges for Monte Carlo sampling
        try:
            house_lo = float(self.mc_range_vars["house_growth_lo"].get())
            house_hi = float(self.mc_range_vars["house_growth_hi"].get())
            rent_lo = float(self.mc_range_vars["rent_growth_lo"].get())
            rent_hi = float(self.mc_range_vars["rent_growth_hi"].get())
            infl_lo = float(self.mc_range_vars["inflation_lo"].get())
            infl_hi = float(self.mc_range_vars["inflation_hi"].get())
            rate_lo = float(self.mc_range_vars["rate_lo"].get())
            rate_hi = float(self.mc_range_vars["rate_hi"].get())
            void_lo = float(self.mc_range_vars["void_lo"].get())
            void_hi = float(self.mc_range_vars["void_hi"].get())
            mgmt_lo = float(self.mc_range_vars["mgmt_lo"].get())
            mgmt_hi = float(self.mc_range_vars["mgmt_hi"].get())
            capex_lo = float(self.mc_range_vars["capex_lo"].get())
            capex_hi = float(self.mc_range_vars["capex_hi"].get())
            maint_prob_lo = float(self.mc_range_vars["maint_prob_lo"].get())
            maint_prob_hi = float(self.mc_range_vars["maint_prob_hi"].get())
            maint_size_lo = float(self.mc_range_vars["maint_size_lo"].get())
            maint_size_hi = float(self.mc_range_vars["maint_size_hi"].get())
        except Exception:
            messagebox.showerror("Input error", "Uncertainty ranges must be numeric.")
            return
        # Limit MC samples to a manageable number for performance
        mc_n = max(0, min(mc_n, 1000))
        results_by_year: Dict[int, List[float]] = {y: [] for y in range(1, base.max_years + 1)}
        rng = random.Random(mc_seed)
        for i in range(mc_n):
            # Clone base params for each simulation
            # Clone the base parameters into a mutable dict; use dataclasses.asdict to avoid relying on model.asdict
            p_kwargs = dict(asdict(base))
            # Sample uncertain parameters using triangular distributions based on user-defined ranges
            p_kwargs['house_price_growth_pct'] = model.triangular_from_95(base.house_price_growth_pct, house_lo, house_hi, rng)
            p_kwargs['rent_growth_pct'] = model.triangular_from_95(base.rent_growth_pct, rent_lo, rent_hi, rng)
            p_kwargs['inflation_pct'] = model.triangular_from_95(base.inflation_pct, infl_lo, infl_hi, rng)
            p_kwargs['annual_rate_pct'] = model.triangular_from_95(base.annual_rate_pct, rate_lo, rate_hi, rng)
            p_kwargs['void_bad_debt_frac'] = model.triangular_from_95(base.void_bad_debt_frac, void_lo, void_hi, rng)
            p_kwargs['letting_mgmt_fee_frac'] = model.triangular_from_95(base.letting_mgmt_fee_frac, mgmt_lo, mgmt_hi, rng)
            p_kwargs['capex_reserve_frac'] = model.triangular_from_95(base.capex_reserve_frac, capex_lo, capex_hi, rng)
            p_kwargs['maint_shock_prob_per_year'] = model.triangular_from_95(base.maint_shock_prob_per_year, maint_prob_lo, maint_prob_hi, rng)
            p_kwargs['maint_shock_size_months_rent'] = model.triangular_from_95(base.maint_shock_size_months_rent, maint_size_lo, maint_size_hi, rng)
            # Do not sample custom interest_rate_path; keep the deterministic path if provided
            p_kwargs['interest_rate_path'] = base.interest_rate_path
            p = model.Params(**p_kwargs)
            out_sample = model.simulate(p, rng=rng)
            for row in out_sample['timeline']:
                results_by_year[row['year']].append(row['profit_if_sold_nominal'])
        # Compute percentile bands and median
        years = list(results_by_year.keys())
        lower_band = []
        median_band = []
        upper_band = []
        for y in years:
            vals = sorted(results_by_year[y])
            if not vals:
                lower_band.append(None)
                median_band.append(None)
                upper_band.append(None)
                continue
            n = len(vals)
            lo_idx = int(round((n - 1) * 0.05))
            med_idx = int(round((n - 1) * 0.50))
            hi_idx = int(round((n - 1) * 0.95))
            lower_band.append(vals[lo_idx])
            median_band.append(vals[med_idx])
            upper_band.append(vals[hi_idx])
        # Prepare input and output summaries for the table windows
        input_map = asdict(base)
        # Use a shallow copy so we can pop off fields without affecting the dataclass
        output_summary: Dict[str, object] = {
            "Initial cash out": f"£{out['initial_cash_out']:,.0f}",
            f"IRR (nominal) sale year {base.sale_year}": f"{out['irr_nominal_sale_year']:.4f}",
        }
        if personal_irr_nominal is not None:
            output_summary[f"Personal IRR (nominal) sale year {base.sale_year}"] = f"{personal_irr_nominal:.4f}"
        worst_dscr = out['worst_dscr_to_sale']
        if worst_dscr is not None and not (isinstance(worst_dscr, float) and worst_dscr == float("inf")):
            output_summary["Worst DSCR to sale_year"] = f"{worst_dscr:.2f}"

        # Create rows for the Ltd table (nominal and real profits)
        ltd_rows = []
        for row in out["timeline"]:
            ltd_rows.append({
                "year": row["year"],
                "nominal": row["profit_if_sold_nominal"],
                "real": row["profit_if_sold_real"],
            })

        # Personal table rows (nominal and real).  If no personal profits, leave empty.
        personal_rows: List[Dict[str, object]] = []
        if personal_profit_if_sold_nominal:
            # Compute real profits for personal series using inflation for each year
            personal_real = []
            for idx, nom in enumerate(personal_profit_if_sold_nominal, start=1):
                # Discount to real at t0 using inflation_pct
                personal_real.append(nom / ((1 + base.inflation_pct / 100.0) ** idx))
            for y_idx, (nom, real) in enumerate(zip(personal_profit_if_sold_nominal, personal_real), start=1):
                personal_rows.append({
                    "year": y_idx,
                    "nominal": nom,
                    "real": real,
                })

        # Update text output with summary only
        self.output_text.delete("1.0", tk.END)
        for k, v in output_summary.items():
            self.output_text.insert(tk.END, f"{k}: {v}\n")
        self.output_text.insert(tk.END, "\nOpen result tables for per‑year profit details.\n")

        # Open tables in new windows
        open_table_window("Ltd – Profit if Sold", ltd_rows, input_map, output_summary)
        if personal_rows:
            open_table_window("Personal – Profit if Sold", personal_rows, input_map, output_summary)

        # Update chart
        self.ax.clear()
        # Dark theme for axes
        self.ax.set_facecolor(PANEL)
        self.ax.tick_params(colors=FG)
        self.ax.spines["bottom"].set_color(FG)
        self.ax.spines["top"].set_color(FG)
        self.ax.spines["left"].set_color(FG)
        self.ax.spines["right"].set_color(FG)
        # Plot deterministic profit path
        years_num = [r['year'] for r in out['timeline']]
        profits_nom = [r['profit_if_sold_nominal'] for r in out['timeline']]
        line_nominal, = self.ax.plot(
            years_num,
            profits_nom,
            label='Nominal profit',
            color=ACCENT,
            linewidth=2,
        )
        # Plot personal path if enabled
        line_personal = None
        if personal_profit_if_sold_nominal:
            years_p = years_num[: len(personal_profit_if_sold_nominal)]
            line_personal, = self.ax.plot(
                years_p,
                personal_profit_if_sold_nominal,
                label='Personal profit',
                color="#10b981",  # teal accent for personal series
                linewidth=2,
            )
        # Shade between 5th and 95th percentiles
        if lower_band and upper_band:
            self.ax.fill_between(
                years,
                lower_band,
                upper_band,
                color="#4b5563",  # muted grey for uncertainty band
                alpha=0.5,
                label='5–95% MC band',
            )
            # Plot median line
            self.ax.plot(years, median_band, color="#9ca3af", linestyle='--', label='MC median')
        self.ax.set_xlabel('Year', color=FG)
        self.ax.set_ylabel('Profit if sold (£)', color=FG)
        self.ax.set_title('Profit trajectory with uncertainty bands', color=FG)
        self.ax.legend(loc="best", facecolor=PANEL, edgecolor=GRID, labelcolor=FG)
        self.figure.tight_layout()

        # Prepare interactive annotation for hover on the lines.  The annotation and
        # callback must be created after the canvas exists (pop‑out window).
        def setup_interactive_annotations():
            # Remove existing callbacks if any
            try:
                self.cid_move
            except AttributeError:
                self.cid_move = None
            if self.cid_move and self.canvas:
                self.canvas.mpl_disconnect(self.cid_move)
            # Create an annotation box hidden by default
            annot = self.ax.annotate(
                "",
                xy=(0, 0),
                xytext=(15, 15),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc=PANEL, ec=ACCENT),
                color=FG,
            )
            annot.set_visible(False)
            def update_annot(line, ind):
                x_data, y_data = line.get_data()
                idx = ind["ind"][0]
                x, y = x_data[idx], y_data[idx]
                annot.xy = (x, y)
                annot.set_text(f"Year {int(x)}\n£{y:,.0f}")
                annot.get_bbox_patch().set(fc=PANEL, ec=ACCENT)
            def on_move(event):
                vis = annot.get_visible()
                if event.inaxes == self.ax:
                    cont_nom, ind_nom = line_nominal.contains(event)
                    if cont_nom:
                        update_annot(line_nominal, ind_nom)
                        annot.set_visible(True)
                        self.canvas.draw_idle()
                        return
                    if line_personal is not None:
                        cont_per, ind_per = line_personal.contains(event)
                        if cont_per:
                            update_annot(line_personal, ind_per)
                            annot.set_visible(True)
                            self.canvas.draw_idle()
                            return
                if vis:
                    annot.set_visible(False)
                    self.canvas.draw_idle()
            # Register the motion notify callback
            if self.canvas:
                self.cid_move = self.canvas.mpl_connect('motion_notify_event', on_move)
        # Finalise layout of figure
        self.figure.tight_layout()
        # Display the graph in a pop‑out window and draw it
        self.show_graph()
        # Set up interactive annotations on the new canvas
        setup_interactive_annotations()


def main() -> None:
    root = tk.Tk()
    app = InvestmentApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
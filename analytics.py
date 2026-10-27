"""Data loading, joining and metric helpers for the Brand Traders dashboard.

Everything here is plain pandas (no Streamlit / Plotly) so it can be unit-tested.

Relational model
----------------
order_details.orderID    -> orders.orderID          (many : 1)
orders.customerID        -> customers.customerID    (many : 1)
orders.employeeID        -> employees.employeeID    (many : 1)
orders.shipperID         -> shippers.shipperID      (many : 1)
order_details.productID  -> products.productID      (many : 1)
products.categoryID      -> categories.categoryID   (many : 1)
employees.reportsTo      -> employees.employeeID    (self reference)
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = [
    "categories", "customers", "employees", "order_details",
    "orders", "products", "shippers",
]

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Customer countries -> ISO-3 (for the choropleth map)
ISO3 = {
    "Argentina": "ARG", "Austria": "AUT", "Belgium": "BEL", "Brazil": "BRA",
    "Canada": "CAN", "Denmark": "DNK", "Finland": "FIN", "France": "FRA",
    "Germany": "DEU", "Ireland": "IRL", "Italy": "ITA", "Mexico": "MEX",
    "Norway": "NOR", "Poland": "POL", "Portugal": "PRT", "Spain": "ESP",
    "Sweden": "SWE", "Switzerland": "CHE", "UK": "GBR", "USA": "USA",
    "Venezuela": "VEN",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def read_csv_any(src) -> pd.DataFrame:
    """Read a CSV from a path or raw bytes, tolerating non-UTF-8 files
    (customers.csv / products.csv are Latin-1 encoded)."""
    last_err = None
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            if isinstance(src, (bytes, bytearray)):
                return pd.read_csv(io.BytesIO(src), encoding=enc)
            return pd.read_csv(src, encoding=enc)
        except UnicodeDecodeError as err:
            last_err = err
    raise last_err  # pragma: no cover


def find_data_dir(candidates) -> str | None:
    """Return the first directory that contains all 7 CSV files."""
    for c in candidates:
        p = Path(c)
        if p.is_dir() and all((p / f"{t}.csv").exists() for t in TABLES):
            return str(p)
    return None


def load_tables(data_dir: str | None, uploads: dict[str, bytes] | None = None) -> dict:
    """Load the 7 tables. Uploaded files override files on disk."""
    uploads = uploads or {}
    tables, missing = {}, []
    for name in TABLES:
        if name in uploads:
            tables[name] = read_csv_any(uploads[name])
        elif data_dir and (Path(data_dir) / f"{name}.csv").exists():
            tables[name] = read_csv_any(str(Path(data_dir) / f"{name}.csv"))
        else:
            missing.append(f"{name}.csv")
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(missing))
    return tables


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #
def validate(t: dict) -> pd.DataFrame:
    """Referential-integrity and null checks across the tables."""
    def orphans(child, ckey, parent, pkey):
        c = t[child][ckey].dropna()
        return int((~c.isin(t[parent][pkey])).sum())

    rows = [
        ("order_details.orderID → orders", orphans("order_details", "orderID", "orders", "orderID")),
        ("order_details.productID → products", orphans("order_details", "productID", "products", "productID")),
        ("orders.customerID → customers", orphans("orders", "customerID", "customers", "customerID")),
        ("orders.employeeID → employees", orphans("orders", "employeeID", "employees", "employeeID")),
        ("orders.shipperID → shippers", orphans("orders", "shipperID", "shippers", "shipperID")),
        ("products.categoryID → categories", orphans("products", "categoryID", "categories", "categoryID")),
        ("employees.reportsTo → employees", orphans("employees", "reportsTo", "employees", "employeeID")),
    ]
    out = pd.DataFrame(rows, columns=["Relationship", "Orphan rows"])
    out["Status"] = np.where(out["Orphan rows"] == 0, "✅ OK", "⚠️ Check")
    return out


def table_overview(t: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [(n, len(df), df.shape[1], int(df.isna().sum().sum())) for n, df in t.items()],
        columns=["Table", "Rows", "Columns", "Null cells"],
    )


# --------------------------------------------------------------------------- #
# Join everything into one analysis table (grain = one order line)
# --------------------------------------------------------------------------- #
def build_master(t: dict) -> pd.DataFrame:
    od = t["order_details"].copy()
    od["discount"] = od["discount"].astype(float)

    o = t["orders"].copy()
    for c in ("orderDate", "requiredDate", "shippedDate"):
        o[c] = pd.to_datetime(o[c], errors="coerce")

    cust = t["customers"].rename(columns={
        "companyName": "customerName", "city": "customerCity", "country": "customerCountry"})

    emp = t["employees"].rename(columns={
        "title": "employeeTitle", "city": "employeeCity",
        "country": "employeeCountry", "reportsTo": "managerID"})
    mgr = emp[["employeeID", "employeeName"]].rename(
        columns={"employeeID": "managerID", "employeeName": "managerName"})
    emp = emp.merge(mgr, on="managerID", how="left")

    ship = t["shippers"].rename(columns={"companyName": "shipperName"})
    prod = t["products"].rename(columns={"unitPrice": "listPrice"})
    cat = t["categories"].rename(columns={"description": "categoryDescription"})

    m = (
        od.merge(o, on="orderID", how="left", validate="m:1")
        .merge(cust, on="customerID", how="left", validate="m:1")
        .merge(emp, on="employeeID", how="left", validate="m:1")
        .merge(ship, on="shipperID", how="left", validate="m:1")
        .merge(prod, on="productID", how="left", validate="m:1")
        .merge(cat, on="categoryID", how="left", validate="m:1")
    )

    # --- measures
    m["grossSales"] = m["unitPrice"] * m["quantity"]
    m["revenue"] = m["grossSales"] * (1 - m["discount"])
    m["discountAmount"] = m["grossSales"] - m["revenue"]

    # --- time
    m["month"] = m["orderDate"].dt.to_period("M").dt.to_timestamp()
    m["year"] = m["orderDate"].dt.year
    m["quarter"] = m["year"].astype(str) + "-Q" + m["orderDate"].dt.quarter.astype(str)
    m["weekday"] = pd.Categorical(m["orderDate"].dt.day_name(), categories=WEEKDAYS, ordered=True)

    # --- fulfilment
    m["isShipped"] = m["shippedDate"].notna()
    m["shipDays"] = (m["shippedDate"] - m["orderDate"]).dt.days
    m["daysVsRequired"] = (m["shippedDate"] - m["requiredDate"]).dt.days  # >0 means late
    m["isLate"] = m["daysVsRequired"].fillna(0) > 0
    return m


def order_level(m: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per order (so freight is not double counted)."""
    first = ["orderDate", "month", "customerID", "customerName", "customerCountry",
             "employeeID", "employeeName", "shipperName", "freight", "shippedDate",
             "requiredDate", "shipDays", "daysVsRequired", "isLate", "isShipped"]
    g = m.groupby("orderID", as_index=False).agg(
        **{c: (c, "first") for c in first},
        revenue=("revenue", "sum"),
        grossSales=("grossSales", "sum"),
        units=("quantity", "sum"),
        lines=("productID", "count"),
    )
    return g


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #
def kpis(m: pd.DataFrame) -> dict:
    if m.empty:
        return dict(revenue=0.0, orders=0, customers=0, units=0, aov=0.0,
                    discount_rate=0.0, freight=0.0, late_rate=0.0)
    o = order_level(m)
    shipped = o[o["isShipped"]]
    return dict(
        revenue=float(m["revenue"].sum()),
        orders=int(o.shape[0]),
        customers=int(o["customerID"].nunique()),
        units=int(m["quantity"].sum()),
        aov=float(m["revenue"].sum() / o.shape[0]),
        discount_rate=float(m["discountAmount"].sum() / m["grossSales"].sum()) if m["grossSales"].sum() else 0.0,
        freight=float(o["freight"].sum()),
        late_rate=float(shipped["isLate"].mean()) if len(shipped) else 0.0,
    )


def pct_change(cur: float, prev: float):
    if prev is None or pd.isna(prev) or prev == 0:
        return None
    return (cur - prev) / prev * 100


def period_agg(m: pd.DataFrame, freq: str = "MS") -> pd.DataFrame:
    """Revenue / orders / units / AOV by period. freq: 'W-MON', 'MS', 'QS'."""
    g = m.groupby(pd.Grouper(key="orderDate", freq=freq)).agg(
        revenue=("revenue", "sum"),
        orders=("orderID", "nunique"),
        units=("quantity", "sum"),
    )
    g["aov"] = g["revenue"] / g["orders"].replace(0, np.nan)
    return g.reset_index().rename(columns={"orderDate": "period"})


def pareto(m: pd.DataFrame, by: str = "productName") -> pd.DataFrame:
    g = m.groupby(by, as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
    g["share"] = g["revenue"] / g["revenue"].sum()
    g["cum_share"] = g["share"].cumsum()
    g["abc"] = np.where(g["cum_share"] - g["share"] < 0.80, "A",
                        np.where(g["cum_share"] - g["share"] < 0.95, "B", "C"))
    return g.reset_index(drop=True)


DISCOUNT_BINS = [-0.001, 0.0, 0.05, 0.10, 0.15, 0.20, 1.0]
DISCOUNT_LABELS = ["No discount", "1–5%", "6–10%", "11–15%", "16–20%", "20%+"]


def discount_analysis(m: pd.DataFrame) -> pd.DataFrame:
    b = pd.cut(m["discount"], bins=DISCOUNT_BINS, labels=DISCOUNT_LABELS)
    g = m.assign(bucket=b).groupby("bucket", observed=True).agg(
        lines=("orderID", "count"),
        avg_quantity=("quantity", "mean"),
        revenue=("revenue", "sum"),
        discount_given=("discountAmount", "sum"),
    ).reset_index()
    g["revenue_share"] = g["revenue"] / g["revenue"].sum()
    return g


def rfm(m: pd.DataFrame) -> pd.DataFrame:
    """Recency / Frequency / Monetary segmentation (quartile scores)."""
    ref = m["orderDate"].max() + pd.Timedelta(days=1)
    g = m.groupby(["customerID", "customerName", "customerCountry"], as_index=False).agg(
        last_order=("orderDate", "max"),
        frequency=("orderID", "nunique"),
        monetary=("revenue", "sum"),
    )
    g["recency"] = (ref - g["last_order"]).dt.days
    if len(g) >= 4:
        g["R"] = pd.qcut(g["recency"].rank(method="first", ascending=False), 4, labels=[1, 2, 3, 4]).astype(int)
        g["F"] = pd.qcut(g["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
        g["M"] = pd.qcut(g["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
        g["score"] = g["R"] + g["F"] + g["M"]
        g["segment"] = np.select(
            [g["score"] >= 10, g["score"] >= 8, g["score"] >= 6],
            ["Champions", "Loyal", "Needs attention"], default="At risk")
    else:
        g["R"] = g["F"] = g["M"] = g["score"] = np.nan
        g["segment"] = "n/a"
    return g.sort_values("monetary", ascending=False).reset_index(drop=True)


def new_customers_by_month(m_all: pd.DataFrame, months: pd.DatetimeIndex) -> pd.DataFrame:
    """New customers per month, using first-ever order date from the full data."""
    first = m_all.groupby("customerID")["orderDate"].min().dt.to_period("M").dt.to_timestamp()
    cnt = first.value_counts().reindex(months, fill_value=0).sort_index()
    return cnt.rename_axis("month").reset_index(name="new_customers")


def employee_summary(m: pd.DataFrame) -> pd.DataFrame:
    o = order_level(m)
    g = o.groupby(["employeeID", "employeeName"], as_index=False).agg(
        orders=("orderID", "count"),
        revenue=("revenue", "sum"),
        customers=("customerID", "nunique"),
        freight=("freight", "sum"),
    )
    disc = m.groupby("employeeID").agg(disc=("discountAmount", "sum"), gross=("grossSales", "sum"))
    g = g.merge((disc["disc"] / disc["gross"]).rename("discount_rate"), on="employeeID", how="left")
    g["aov"] = g["revenue"] / g["orders"]
    g["revenue_share"] = g["revenue"] / g["revenue"].sum()
    info = m.drop_duplicates("employeeID")[["employeeID", "employeeTitle", "employeeCity", "managerName"]]
    return g.merge(info, on="employeeID", how="left").sort_values("revenue", ascending=False)


def shipper_summary(m: pd.DataFrame) -> pd.DataFrame:
    o = order_level(m)
    s = o[o["isShipped"]]
    g = o.groupby("shipperName").agg(
        orders=("orderID", "count"), freight=("freight", "sum"),
        avg_freight=("freight", "mean"), revenue=("revenue", "sum"),
        pending=("isShipped", lambda x: int((~x).sum())),
    )
    g["avg_ship_days"] = s.groupby("shipperName")["shipDays"].mean()
    g["late_rate"] = s.groupby("shipperName")["isLate"].mean()
    g["freight_pct_revenue"] = g["freight"] / g["revenue"]
    return g.reset_index()


def insights(m: pd.DataFrame) -> list[str]:
    """Plain-English headline findings for the filtered data."""
    if m.empty:
        return []
    out = []
    tot = m["revenue"].sum()
    cat = m.groupby("categoryName")["revenue"].sum().sort_values(ascending=False)
    out.append(f"**{cat.index[0]}** is the top category with **${cat.iloc[0]:,.0f}** "
               f"({cat.iloc[0] / tot:.0%} of revenue).")
    prod = m.groupby("productName")["revenue"].sum().sort_values(ascending=False)
    out.append(f"Best-selling product by revenue: **{prod.index[0]}** (${prod.iloc[0]:,.0f}).")
    ctry = m.groupby("customerCountry")["revenue"].sum().sort_values(ascending=False)
    top3 = ctry.head(3)
    out.append(f"Top markets: **{', '.join(top3.index)}** — together {top3.sum() / tot:.0%} of revenue.")
    mon = m.groupby("month")["revenue"].sum()
    if len(mon) > 1:
        out.append(f"Peak month: **{mon.idxmax():%B %Y}** (${mon.max():,.0f}).")
    emp = m.groupby("employeeName")["revenue"].sum().sort_values(ascending=False)
    out.append(f"Top salesperson: **{emp.index[0]}** (${emp.iloc[0]:,.0f}).")
    cust = m.groupby("customerName")["revenue"].sum().sort_values(ascending=False)
    out.append(f"Top 10 customers generate **{cust.head(10).sum() / tot:.0%}** of revenue.")
    k = kpis(m)
    out.append(f"Average discount rate is **{k['discount_rate']:.1%}**; "
               f"**{k['late_rate']:.1%}** of shipped orders left after the required date.")
    return out

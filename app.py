"""Brand Traders — Sales & Operations Analytics (Streamlit)

Run:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import analytics as A

st.set_page_config(page_title="Brand Traders Analytics", page_icon="📊", layout="wide")

PALETTE = px.colors.qualitative.Safe
BLUE = "#2563eb"


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def _stretch(fn, *args, **kwargs):
    """Full-width rendering that works across Streamlit versions."""
    try:
        return fn(*args, width="stretch", **kwargs)
    except TypeError:
        return fn(*args, use_container_width=True, **kwargs)


def show(fig, height: int = 380):
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        colorway=PALETTE, legend=dict(title=None),
    )
    _stretch(st.plotly_chart, fig)


def table(df: pd.DataFrame, **kwargs):
    _stretch(st.dataframe, df, hide_index=True, **kwargs)


def money(v: float) -> str:
    return f"${v:,.0f}"


def short_money(v: float) -> str:
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:.1f}K"
    return f"${v:,.0f}"


def metric(col, label, value, prev=None, cur=None, invert=False):
    delta = A.pct_change(cur, prev) if (cur is not None and prev is not None) else None
    col.metric(label, value,
               f"{delta:+.1f}% vs prior period" if delta is not None else None,
               delta_color="inverse" if invert else "normal")


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading and joining tables…")
def get_data(data_dir, uploads: tuple):
    tables = A.load_tables(data_dir, dict(uploads))
    return tables, A.build_master(tables), A.validate(tables), A.table_overview(tables)


here = Path(__file__).parent
data_dir = A.find_data_dir([here / "data", here, Path("data"), Path("."),
                            "/mnt/user-data/uploads"])

with st.sidebar:
    st.title("📊 Brand Traders")
    st.caption("Sales & operations analytics")
    with st.expander("📁 Data source", expanded=data_dir is None):
        st.caption("Reads the 7 CSVs from `./data` automatically. "
                   "Or upload them here (uploads take priority).")
        files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
    uploads = tuple(sorted((Path(f.name).stem, f.getvalue()) for f in files)) if files else ()

try:
    tables, M, checks, overview = get_data(data_dir, uploads)
except FileNotFoundError as err:
    st.error(f"{err}. Put the CSV files in a `data/` folder next to `app.py`, "
             "or upload them from the sidebar.")
    st.stop()

# --------------------------------------------------------------------------- #
# Sidebar filters
# --------------------------------------------------------------------------- #
dmin, dmax = M["orderDate"].min().date(), M["orderDate"].max().date()

with st.sidebar:
    st.subheader("Filters")
    picked = st.date_input("Order date range", (dmin, dmax), min_value=dmin, max_value=dmax)
    d0, d1 = (picked if isinstance(picked, (tuple, list)) and len(picked) == 2 else (dmin, dmax))

    countries = sorted(M["customerCountry"].dropna().unique())
    f_country = st.multiselect("Customer country", countries, default=countries)
    cats = sorted(M["categoryName"].dropna().unique())
    f_cat = st.multiselect("Product category", cats, default=cats)
    emps = sorted(M["employeeName"].dropna().unique())
    f_emp = st.multiselect("Salesperson", emps, default=emps)
    ships = sorted(M["shipperName"].dropna().unique())
    f_ship = st.multiselect("Shipper", ships, default=ships)
    f_disc = st.checkbox("Include discontinued products", value=True)

base = M[
    M["customerCountry"].isin(f_country) & M["categoryName"].isin(f_cat)
    & M["employeeName"].isin(f_emp) & M["shipperName"].isin(f_ship)
]
if not f_disc:
    base = base[base["discontinued"] == 0]

d0, d1 = pd.Timestamp(d0), pd.Timestamp(d1)
F = base[base["orderDate"].between(d0, d1)]

span = (d1 - d0).days + 1
prev_F = base[base["orderDate"].between(d0 - pd.Timedelta(days=span), d0 - pd.Timedelta(days=1))]

if F.empty:
    st.warning("No data matches the current filters. Widen the date range or selections.")
    st.stop()

with st.sidebar:
    st.divider()
    st.caption(f"**{F['orderID'].nunique():,}** orders · **{len(F):,}** order lines in view")
    st.caption("Freight is an order-level cost; it is counted once per order.")

O = A.order_level(F)  # one row per order
cur, prv = A.kpis(F), (A.kpis(prev_F) if not prev_F.empty else None)

st.title("Brand Traders — Sales & Operations Analytics")
st.caption(f"{d0:%d %b %Y} → {d1:%d %b %Y} · all 7 tables joined "
           "(order lines ⨝ orders ⨝ customers ⨝ employees ⨝ shippers ⨝ products ⨝ categories)")

tabs = st.tabs(["🏠 Overview", "📈 Sales trends", "📦 Products", "👥 Customers",
                "🧑‍💼 Employees", "🚚 Shipping", "🌍 Geography", "🗂️ Data & model"])

# =========================================================================== #
# 1. OVERVIEW
# =========================================================================== #
with tabs[0]:
    c = st.columns(4)
    g = (lambda k: prv[k] if prv else None)
    metric(c[0], "Revenue", short_money(cur["revenue"]), g("revenue"), cur["revenue"])
    metric(c[1], "Orders", f"{cur['orders']:,}", g("orders"), cur["orders"])
    metric(c[2], "Avg order value", money(cur["aov"]), g("aov"), cur["aov"])
    metric(c[3], "Active customers", f"{cur['customers']:,}", g("customers"), cur["customers"])
    c = st.columns(4)
    metric(c[0], "Units sold", f"{cur['units']:,}", g("units"), cur["units"])
    metric(c[1], "Avg discount rate", f"{cur['discount_rate']:.1%}")
    metric(c[2], "Freight cost", short_money(cur["freight"]), g("freight"), cur["freight"], invert=True)
    metric(c[3], "Late shipments", f"{cur['late_rate']:.1%}")
    if prv is None:
        st.caption("Prior-period comparison appears when an earlier window of the same length exists in the data.")

    left, right = st.columns([3, 2])
    with left:
        mo = A.period_agg(F, "MS")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=mo["period"], y=mo["revenue"], name="Revenue", marker_color=BLUE, opacity=0.85)
        fig.add_scatter(x=mo["period"], y=mo["orders"], name="Orders", mode="lines+markers",
                        line=dict(color="#f97316", width=2), secondary_y=True)
        fig.update_layout(title="Monthly revenue & order count")
        fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
        fig.update_yaxes(title_text="Orders", secondary_y=True)
        show(fig)
        st.caption("First and last months may be partial depending on the selected dates.")
    with right:
        cat = F.groupby("categoryName", as_index=False)["revenue"].sum()
        fig = px.pie(cat, names="categoryName", values="revenue", hole=0.5, title="Revenue by category")
        fig.update_traces(textinfo="percent+label", textposition="outside")
        fig.update_layout(showlegend=False)
        show(fig)

    st.subheader("Key insights")
    for line in A.insights(F):
        st.markdown(f"- {line}")

# =========================================================================== #
# 2. SALES TRENDS
# =========================================================================== #
with tabs[1]:
    c1, c2, c3 = st.columns(3)
    gran = c1.radio("Granularity", ["Weekly", "Monthly", "Quarterly"], index=1, horizontal=True)
    measure = c2.selectbox("Measure", ["Revenue", "Orders", "Units", "Avg order value"])
    win = c3.slider("Moving-average window (periods)", 1, 6, 3)
    freq = {"Weekly": "W-MON", "Monthly": "MS", "Quarterly": "QS"}[gran]
    col = {"Revenue": "revenue", "Orders": "orders", "Units": "units", "Avg order value": "aov"}[measure]

    P = A.period_agg(F, freq)
    P["moving_avg"] = P[col].rolling(win, min_periods=1).mean()
    P["growth_pct"] = P[col].pct_change() * 100

    fig = go.Figure()
    fig.add_bar(x=P["period"], y=P[col], name=measure, marker_color=BLUE, opacity=0.6)
    fig.add_scatter(x=P["period"], y=P["moving_avg"], name=f"{win}-period moving avg",
                    line=dict(color="#dc2626", width=3))
    fig.update_layout(title=f"{measure} — {gran.lower()}")
    show(fig, 420)

    a, b = st.columns(2)
    with a:
        G = P.dropna(subset=["growth_pct"])
        G = G[np.isfinite(G["growth_pct"])]
        fig = px.bar(G, x="period", y="growth_pct", title=f"{gran} growth in {measure.lower()} (%)",
                     color=(G["growth_pct"] >= 0).map({True: "Growth", False: "Decline"}),
                     color_discrete_map={"Growth": "#16a34a", "Decline": "#dc2626"})
        fig.update_layout(showlegend=False)
        show(fig)
    with b:
        hm = F.pivot_table(index="year", columns=F["orderDate"].dt.month, values="revenue", aggfunc="sum")
        hm.columns = [pd.Timestamp(2000, int(mth), 1).strftime("%b") for mth in hm.columns]
        fig = px.imshow(hm, text_auto=".2s", aspect="auto", color_continuous_scale="Blues",
                        title="Revenue heatmap — year × month")
        fig.update_yaxes(type="category")
        show(fig)

    a, b = st.columns(2)
    with a:
        wd = F.groupby("weekday", observed=False)["revenue"].sum().reindex(A.WEEKDAYS).reset_index()
        show(px.bar(wd, x="weekday", y="revenue", title="Revenue by weekday", color_discrete_sequence=[BLUE]))
    with b:
        cm = F.groupby(["month", "categoryName"], as_index=False)["revenue"].sum()
        show(px.area(cm, x="month", y="revenue", color="categoryName", title="Category mix over time"))

    cumu = A.period_agg(F, "MS")
    cumu["cumulative_revenue"] = cumu["revenue"].cumsum()
    show(px.area(cumu, x="period", y="cumulative_revenue", title="Cumulative revenue",
                 color_discrete_sequence=[BLUE]), 300)

# =========================================================================== #
# 3. PRODUCTS
# =========================================================================== #
with tabs[2]:
    c1, c2 = st.columns([1, 2])
    topn = c1.slider("Top N products", 5, 30, 10)
    pmeasure = c2.radio("Rank by", ["Revenue", "Units sold"], horizontal=True)
    pcol = "revenue" if pmeasure == "Revenue" else "quantity"

    pr = (F.groupby(["productName", "categoryName"], as_index=False)
            .agg(revenue=("revenue", "sum"), quantity=("quantity", "sum")))
    top = pr.sort_values(pcol, ascending=False).head(topn).iloc[::-1]
    a, b = st.columns(2)
    with a:
        show(px.bar(top, x=pcol, y="productName", color="categoryName", orientation="h",
                    title=f"Top {topn} products by {pmeasure.lower()}"), 90 + 28 * topn)
    with b:
        show(px.treemap(F, path=[px.Constant("All"), "categoryName", "productName"],
                        values="revenue", title="Revenue by category → product"), 90 + 28 * topn)

    a, b = st.columns(2)
    with a:
        P_ = A.pareto(F)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=np.arange(1, len(P_) + 1), y=P_["revenue"], name="Revenue",
                    marker_color=P_["abc"].map({"A": "#2563eb", "B": "#f59e0b", "C": "#9ca3af"}),
                    hovertext=P_["productName"])
        fig.add_scatter(x=np.arange(1, len(P_) + 1), y=P_["cum_share"] * 100, name="Cumulative %",
                        line=dict(color="#dc2626"), secondary_y=True)
        fig.update_layout(title="Pareto (ABC): products ranked by revenue", xaxis_title="Product rank")
        fig.update_yaxes(title_text="Cumulative % of revenue", range=[0, 105], secondary_y=True)
        show(fig)
        n = P_["abc"].value_counts()
        st.caption(f"**A** ({n.get('A', 0)} products) drive ~80% of revenue · "
                   f"**B** ({n.get('B', 0)}) next 15% · **C** ({n.get('C', 0)}) last 5%.")
    with b:
        ps = F.groupby(["productName", "categoryName", "listPrice"], as_index=False).agg(
            units=("quantity", "sum"), revenue=("revenue", "sum"))
        show(px.scatter(ps, x="listPrice", y="units", size="revenue", color="categoryName",
                        hover_name="productName", title="Price vs. volume (bubble = revenue)",
                        size_max=40, labels={"listPrice": "List price ($)"}))

    a, b = st.columns(2)
    with a:
        D = A.discount_analysis(F)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=D["bucket"].astype(str), y=D["lines"], name="Order lines", marker_color=BLUE)
        fig.add_scatter(x=D["bucket"].astype(str), y=D["avg_quantity"], name="Avg qty / line",
                        line=dict(color="#f97316", width=3), secondary_y=True)
        fig.update_layout(title="Discount level vs. order-line volume & quantity")
        fig.update_yaxes(title_text="Avg quantity", secondary_y=True)
        show(fig)
    with b:
        cs = F.groupby("categoryName", as_index=False).agg(
            gross=("grossSales", "sum"), discount=("discountAmount", "sum"))
        cs["discount_rate"] = cs["discount"] / cs["gross"] * 100
        show(px.bar(cs.sort_values("discount_rate"), x="discount_rate", y="categoryName", orientation="h",
                    title="Effective discount rate by category (%)", color_discrete_sequence=["#f97316"]))

    st.subheader("Product table")
    pt = F.groupby(["productName", "categoryName", "discontinued"], as_index=False).agg(
        revenue=("revenue", "sum"), units=("quantity", "sum"),
        orders=("orderID", "nunique"), avg_discount=("discount", "mean"))
    pt["avg_discount"] = pt["avg_discount"] * 100
    pt["discontinued"] = pt["discontinued"].map({0: "Active", 1: "Discontinued"})
    table(pt.sort_values("revenue", ascending=False), column_config={
        "revenue": st.column_config.NumberColumn("Revenue", format="$%.0f"),
        "avg_discount": st.column_config.NumberColumn("Avg discount (%)", format="%.1f%%"),
    })

# =========================================================================== #
# 4. CUSTOMERS
# =========================================================================== #
with tabs[3]:
    cust = (F.groupby(["customerName", "customerCountry"], as_index=False)
              .agg(revenue=("revenue", "sum"), orders=("orderID", "nunique")))
    k1, k2, k3 = st.columns(3)
    k1.metric("Customers in view", f"{len(cust):,}")
    k2.metric("Avg revenue / customer", money(cust["revenue"].mean()))
    top10 = cust.nlargest(10, "revenue")["revenue"].sum() / cust["revenue"].sum()
    k3.metric("Top-10 customer share", f"{top10:.0%}")

    a, b = st.columns(2)
    with a:
        n = st.slider("Top N customers", 5, 25, 10, key="topcust")
        t_ = cust.nlargest(n, "revenue").iloc[::-1]
        show(px.bar(t_, x="revenue", y="customerName", color="customerCountry", orientation="h",
                    title=f"Top {n} customers by revenue"), 90 + 28 * n)
    with b:
        show(px.scatter(cust, x="orders", y="revenue", color="customerCountry", hover_name="customerName",
                        title="Order frequency vs. revenue per customer"), 90 + 28 * n)

    a, b = st.columns(2)
    with a:
        months = pd.date_range(F["month"].min(), F["month"].max(), freq="MS")
        NC = A.new_customers_by_month(M, months)
        show(px.bar(NC, x="month", y="new_customers", title="New customers acquired per month",
                    color_discrete_sequence=["#16a34a"]), 320)
    with b:
        ord_per = cust["orders"].value_counts().sort_index().reset_index()
        ord_per.columns = ["orders", "customers"]
        show(px.bar(ord_per, x="orders", y="customers", title="How many orders do customers place?",
                    color_discrete_sequence=[BLUE]), 320)

    st.subheader("RFM segmentation")
    st.caption("Customers scored 1–4 on Recency (days since last order), Frequency (orders) and "
               "Monetary (revenue). Total score ≥10 = Champions, 8–9 = Loyal, 6–7 = Needs attention, ≤5 = At risk.")
    R = A.rfm(F)
    a, b = st.columns([2, 3])
    with a:
        seg = R.groupby("segment", as_index=False).agg(customers=("customerID", "count"),
                                                       revenue=("monetary", "sum"))
        show(px.bar(seg, x="segment", y="revenue", color="segment", text="customers",
                    title="Revenue by segment (label = # customers)",
                    category_orders={"segment": ["Champions", "Loyal", "Needs attention", "At risk"]}), 340)
    with b:
        show(px.scatter(R, x="recency", y="monetary", size="frequency", color="segment",
                        hover_name="customerName", title="Recency vs. monetary (bubble = orders)",
                        labels={"recency": "Days since last order"}), 340)
    table(R[["customerName", "customerCountry", "recency", "frequency", "monetary", "R", "F", "M", "segment"]],
          column_config={"monetary": st.column_config.NumberColumn("Revenue", format="$%.0f")})

# =========================================================================== #
# 5. EMPLOYEES
# =========================================================================== #
with tabs[4]:
    E = A.employee_summary(F)
    a, b = st.columns(2)
    with a:
        show(px.bar(E.sort_values("revenue"), x="revenue", y="employeeName", orientation="h",
                    color="employeeTitle", title="Revenue by salesperson"))
    with b:
        show(px.scatter(E, x="orders", y="aov", size="revenue", color="employeeName",
                        title="Orders handled vs. avg order value (bubble = revenue)",
                        labels={"aov": "Avg order value ($)"}))

    a, b = st.columns(2)
    with a:
        hm = F.pivot_table(index="employeeName", columns="categoryName", values="revenue", aggfunc="sum").fillna(0)
        show(px.imshow(hm, text_auto=".2s", aspect="auto", color_continuous_scale="Blues",
                       title="Salesperson × category revenue"))
    with b:
        em = F.groupby(["month", "employeeName"], as_index=False)["revenue"].sum()
        show(px.line(em, x="month", y="revenue", color="employeeName", title="Monthly revenue by salesperson"))

    st.subheader("Scorecard")
    Esc = E[["employeeName", "employeeTitle", "employeeCity", "managerName", "orders", "customers",
             "revenue", "revenue_share", "aov", "discount_rate", "freight"]].copy()
    Esc["revenue_share"] *= 100
    Esc["discount_rate"] *= 100
    table(Esc,
          column_config={
              "revenue": st.column_config.NumberColumn("Revenue", format="$%.0f"),
              "aov": st.column_config.NumberColumn("Avg order", format="$%.0f"),
              "freight": st.column_config.NumberColumn("Freight", format="$%.0f"),
              "revenue_share": st.column_config.ProgressColumn("Share of revenue", format="%.1f%%", min_value=0,
                                                              max_value=float(Esc["revenue_share"].max())),
              "discount_rate": st.column_config.NumberColumn("Discount rate", format="%.1f%%"),
          })

# =========================================================================== #
# 6. SHIPPING
# =========================================================================== #
with tabs[5]:
    S = A.shipper_summary(F)
    shipped = O[O["isShipped"]]
    c = st.columns(4)
    c[0].metric("Avg days to ship", f"{shipped['shipDays'].mean():.1f}")
    c[1].metric("Late shipments", f"{cur['late_rate']:.1%}")
    c[2].metric("Avg freight / order", money(O["freight"].mean()))
    c[3].metric("Not yet shipped", f"{int((~O['isShipped']).sum()):,}")

    a, b = st.columns(2)
    with a:
        show(px.pie(S, names="shipperName", values="orders", hole=0.5, title="Order share by shipper"))
    with b:
        show(px.bar(S, x="shipperName", y="avg_freight", color="shipperName",
                    title="Average freight per order ($)", text_auto=".2f"))

    a, b = st.columns(2)
    with a:
        show(px.box(shipped, x="shipperName", y="shipDays", color="shipperName",
                    title="Days from order to shipment", points="outliers"))
    with b:
        S2 = S.assign(late_pct=S["late_rate"] * 100)
        show(px.bar(S2, x="shipperName", y="late_pct", color="shipperName", text_auto=".1f",
                    title="Late-shipment rate (%)"))

    a, b = st.columns(2)
    with a:
        show(px.scatter(O, x="revenue", y="freight", color="shipperName", opacity=0.6,
                        title="Freight vs. order value", labels={"revenue": "Order revenue ($)"}))
    with b:
        fm = O.groupby(["customerCountry"], as_index=False).agg(freight=("freight", "sum"), revenue=("revenue", "sum"))
        fm["freight_pct"] = fm["freight"] / fm["revenue"] * 100
        show(px.bar(fm.sort_values("freight_pct", ascending=False).head(12), x="customerCountry", y="freight_pct",
                    title="Freight as % of revenue — top 12 countries", color_discrete_sequence=["#f97316"]))

    st.subheader("Shipper scorecard")
    Ssc = S.copy()
    Ssc["late_rate"] *= 100
    Ssc["freight_pct_revenue"] *= 100
    table(Ssc, column_config={
        "freight": st.column_config.NumberColumn(format="$%.0f"),
        "avg_freight": st.column_config.NumberColumn(format="$%.2f"),
        "revenue": st.column_config.NumberColumn(format="$%.0f"),
        "avg_ship_days": st.column_config.NumberColumn(format="%.1f"),
        "late_rate": st.column_config.NumberColumn("Late rate", format="%.1f%%"),
        "freight_pct_revenue": st.column_config.NumberColumn("Freight % of revenue", format="%.1f%%"),
    })

    pend = O[~O["isShipped"]]
    if len(pend):
        st.subheader(f"Orders not yet shipped ({len(pend)})")
        table(pend[["orderID", "orderDate", "requiredDate", "customerName", "customerCountry",
                    "employeeName", "shipperName", "revenue"]].sort_values("requiredDate"),
              column_config={"revenue": st.column_config.NumberColumn(format="$%.0f")})

# =========================================================================== #
# 7. GEOGRAPHY
# =========================================================================== #
with tabs[6]:
    geo = (F.groupby("customerCountry", as_index=False)
             .agg(revenue=("revenue", "sum"), orders=("orderID", "nunique"), customers=("customerID", "nunique")))
    geo["iso3"] = geo["customerCountry"].map(A.ISO3)
    geo["aov"] = geo["revenue"] / geo["orders"]

    fig = px.choropleth(geo.dropna(subset=["iso3"]), locations="iso3", color="revenue",
                        hover_name="customerCountry", hover_data={"orders": True, "customers": True, "iso3": False},
                        color_continuous_scale="Blues", title="Revenue by customer country")
    fig.update_geos(showframe=False, projection_type="natural earth", fitbounds="locations")
    show(fig, 480)

    a, b = st.columns(2)
    with a:
        show(px.bar(geo.sort_values("revenue"), x="revenue", y="customerCountry", orientation="h",
                    title="Revenue by country", color_discrete_sequence=[BLUE]), 520)
    with b:
        city = F.groupby(["customerCountry", "customerCity"], as_index=False)["revenue"].sum()
        show(px.treemap(city, path=[px.Constant("World"), "customerCountry", "customerCity"], values="revenue",
                        title="Country → city"), 520)

    hm = F.pivot_table(index="customerCountry", columns="categoryName", values="revenue", aggfunc="sum").fillna(0)
    show(px.imshow(hm, text_auto=".2s", aspect="auto", color_continuous_scale="Blues",
                   title="What each country buys — country × category"), 150 + 22 * len(hm))
    table(geo.drop(columns="iso3").sort_values("revenue", ascending=False),
          column_config={"revenue": st.column_config.NumberColumn(format="$%.0f"),
                         "aov": st.column_config.NumberColumn("Avg order", format="$%.0f")})

# =========================================================================== #
# 8. DATA & MODEL
# =========================================================================== #
with tabs[7]:
    st.subheader("How the tables relate")
    st.code(
        "categories ──< products ──< order_details >── orders >── customers\n"
        "                                                │\n"
        "                                                ├──>── employees ──(reportsTo)──> employees\n"
        "                                                └──>── shippers\n\n"
        "order_details.orderID   → orders.orderID\n"
        "order_details.productID → products.productID\n"
        "products.categoryID     → categories.categoryID\n"
        "orders.customerID       → customers.customerID\n"
        "orders.employeeID       → employees.employeeID\n"
        "orders.shipperID        → shippers.shipperID\n"
        "employees.reportsTo     → employees.employeeID",
        language="text")
    st.markdown(
        "**Analysis grain:** one row per *order line* (`order_details`). "
        "`revenue = unitPrice × quantity × (1 − discount)` uses the price actually charged on the order "
        "line, not the product's current list price. Order-level fields such as `freight` are "
        "de-duplicated by order before summing.")

    a, b = st.columns(2)
    with a:
        st.markdown("**Tables loaded**")
        table(overview)
    with b:
        st.markdown("**Referential-integrity checks**")
        table(checks)

    st.subheader("Joined analysis table")
    st.caption(f"{len(F):,} rows × {F.shape[1]} columns (respects the sidebar filters)")
    _stretch(st.dataframe, F.head(500), hide_index=True)
    st.download_button("⬇️ Download joined data (CSV)", F.to_csv(index=False).encode("utf-8"),
                       file_name="brand_traders_joined.csv", mime="text/csv")

    with st.expander("Browse raw tables"):
        name = st.selectbox("Table", A.TABLES)
        _stretch(st.dataframe, tables[name], hide_index=True)

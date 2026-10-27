streamlit app https://brandtraders-tayyabaramzan.streamlit.app/
# Brand Traders – Sales & Operations Analytics

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app reads the 7 CSVs from `./data` automatically. You can also upload them from the sidebar.

## Files
- `app.py`        – Streamlit UI (8 tabs + global sidebar filters)
- `analytics.py`  – loading, table joins, KPI / RFM / Pareto helpers (pure pandas)
- `data/`         – categories, customers, employees, order_details, orders, products, shippers
"# Brand_traders" 

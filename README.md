# -stock-streamlit-app21

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy / host correctly

If you host this as a **static website**, you will only see the file index page (`downloads/index.html`).
This project is a **Streamlit app** and must run as a Python web process.

Use one of these start commands on your hosting platform:

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true
```

or rely on the included `Procfile`:

```text
web: streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true
```

## Main app files

- `app.py` (deployment entrypoint)
- `app_hardened (2).py` (full dashboard implementation)
- `requirements.txt` (deployment dependencies)

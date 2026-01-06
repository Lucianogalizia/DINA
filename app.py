# app.py
# ==========================================================
# STREAMLIT - Interfaz DIN (CS) + NIV
# Versión: Local + Google Cloud Run (Bucket GCS)
#
# IDEA SIMPLE:
# - En LOCAL: lee indices/parquet y data_store como hoy.
# - En CLOUD RUN: lee indices/parquet desde un Bucket (gs://...)
#                y baja los .din necesarios "on-demand" a /tmp
#
# Requisitos:
# - Variable de entorno DINAS_BUCKET (solo en Cloud Run)
#   Ej: DINAS_BUCKET="dinas-data"
# - En el bucket deben existir:
#   gs://DINAS_BUCKET/din_index.parquet
#   gs://DINAS_BUCKET/niv_index.parquet
#   gs://DINAS_BUCKET/data_store/din/...
#   gs://DINAS_BUCKET/data_store/niv/...
# ==========================================================

import os
import re
import tempfile
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ---------- (NUEVO) GCS ----------
GCS_BUCKET = os.environ.get("DINAS_BUCKET", "").strip()  # si está vacío -> modo local
GCS_PREFIX = os.environ.get("DINAS_GCS_PREFIX", "").strip().strip("/")  # opcional, por si querés guardar dentro de una carpeta
# Ej: DINAS_GCS_PREFIX="interfaz_dinas" -> gs://bucket/interfaz_dinas/...

def _gcs_join(*parts: str) -> str:
    parts = [p.strip("/").replace("\\", "/") for p in parts if p is not None and str(p).strip() != ""]
    suffix = "/".join(parts)
    if GCS_PREFIX:
        suffix = f"{GCS_PREFIX}/{suffix}"
    return f"gs://{GCS_BUCKET}/{suffix}"

def _is_gs_path(p: str | None) -> bool:
    return bool(p) and str(p).strip().lower().startswith("gs://")

@st.cache_resource(show_spinner=False)
def _get_gcs_client():
    # Se usa service account por defecto del runtime de Cloud Run
    # (cuando lo configuremos, te digo cómo)
    try:
        from google.cloud import storage
        return storage.Client()
    except Exception as e:
        return None

def _parse_gs_url(gs_url: str):
    # gs://bucket/path/to/file
    u = gs_url.strip()
    if not u.lower().startswith("gs://"):
        raise ValueError("No es gs://")
    u = u[5:]
    bucket, _, blob = u.partition("/")
    return bucket, blob

def _gcs_download_to_temp(gs_url: str) -> str:
    """
    Baja un archivo desde GCS a /tmp y devuelve el path local.
    Cachea por Streamlit (por gs_url) para no bajar siempre.
    """
    client = _get_gcs_client()
    if client is None:
        raise RuntimeError("No está disponible google-cloud-storage. Agregalo al requirements.txt")

    bucket_name, blob_name = _parse_gs_url(gs_url)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Nombre estable en /tmp según blob
    safe_name = blob_name.replace("/", "__")
    local_path = os.path.join(tempfile.gettempdir(), safe_name)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    blob.download_to_filename(local_path)
    return local_path

def _exists_local(p: str | None) -> bool:
    if not p:
        return False
    try:
        return Path(str(p)).exists()
    except Exception:
        return False

# ---------------- CONFIG ----------------
# En LOCAL seguís usando tus paths (si existen).
# En CLOUD, si no existen, usa bucket.
PROYECTO_DIR = r"C:\Users\dgalizia\Desktop\Proyectos de IA\Interfaz Dinas"

INDEX_PARQUET_LOCAL = os.path.join(PROYECTO_DIR, "din_index.parquet")
INDEX_CSV_LOCAL     = os.path.join(PROYECTO_DIR, "din_index.csv")
NIV_INDEX_LOCAL     = os.path.join(PROYECTO_DIR, "niv_index.parquet")

# En bucket (Cloud Run)
INDEX_PARQUET_GCS = _gcs_join("din_index.parquet") if GCS_BUCKET else ""
NIV_INDEX_GCS     = _gcs_join("niv_index.parquet") if GCS_BUCKET else ""

# Roots para resolver paths (LOCAL)
DATA_ROOTS = [
    r"O:\Petroleum\Upstream\Desarrollo Operativo\Mediciones Fisicas",
    PROYECTO_DIR,
]

SECTION_RE = re.compile(r"^\s*\[(.+?)\]\s*$")
KV_RE      = re.compile(r"^\s*([^=]+?)\s*=\s*(.*?)\s*$")
POINT_KEY_RE = re.compile(r"^(X|Y)\s*(\d+)$", re.IGNORECASE)

# Extras DIN (GM -> GPM)
EXTRA_FIELDS = {
    "AIB Carrera": ("AIB", "CS"),
    "Sentido giro": ("AIB", "SG"),
    "Tipo Contrapesos": ("CONTRAPESO", "TP"),
    "Distancia contrapesos (cm)": ("CONTRAPESO", "DE"),
    "Contrapeso actual": ("RARE", "CA"),
    "Contrapeso ideal": ("RARE", "CM"),
    "AIBEB_Torque max contrapeso": ("RAEB", "TM"),
    "%Estructura": ("RARE", "SE"),
    "%Balance": ("RARR", "PC"),
    "Bba Diam Pistón": ("BOMBA", "DP"),
    "Bba Prof": ("BOMBA", "PB"),
    "Bba Llenado": ("BOMBA", "CA"),
    "GPM": ("AIB", "GM"),
    "Polea Motor": ("MOTOR", "DP"),
    "Potencia Motor": ("MOTOR", "PN"),
    "RPM Motor": ("MOTOR", "RM"),
}

# ---------------- Helpers ----------------
def read_text_best_effort(path: Path) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc, errors="strict")
        except Exception:
            pass
    return path.read_text(encoding="latin-1", errors="ignore")

def normalize_no_exact(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    if s.upper() in ("<NA>", "NAN", "NONE"):
        return ""
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r"\s+", "", s)
    s = s.casefold().upper()
    return s

def normalize_fe_date(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (datetime, pd.Timestamp)):
        return pd.to_datetime(x).date()
    s = str(x).strip()
    if not s:
        return None
    dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return dt.date() if not pd.isna(dt) else None

def normalize_ho_str(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    try:
        t = pd.to_datetime(s, errors="coerce").time()
        if t:
            return f"{t.hour:02d}:{t.minute:02d}"
    except Exception:
        pass
    m = re.match(r"^(\d{1,2}):(\d{2})", s)
    if m:
        return f"{int(m.group(1)):02d}:{int(m.group(2)):02d}"
    return s

def safe_to_float(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if not s:
        return None
    if "=" in s:
        s = s.split("=")[-1].strip()
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def find_col(df: pd.DataFrame, candidates):
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    cols_upper = {c.upper(): c for c in cols}
    for cand in candidates:
        if cand.upper() in cols_upper:
            return cols_upper[cand.upper()]
    return None

def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    seen = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 1
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}_dup{seen[c]}")
    df.columns = new_cols
    return df

# ---------- (NUEVO) Mapear paths locales a GCS ----------
def map_local_datastore_to_gcs(path_str: str | None) -> str | None:
    """
    Si el índice trae un path local como:
      C:\...\Interfaz Dinas\data_store\din\Pozo\2025-12\archivo.din
    lo convertimos a:
      gs://DINAS_BUCKET/data_store/din/Pozo/2025-12/archivo.din
    """
    if not path_str or not GCS_BUCKET:
        return None

    p = str(path_str).replace("\\", "/")
    idx = p.lower().find("/data_store/")
    if idx == -1:
        return None

    rel = p[idx+1:]  # "data_store/..."
    return _gcs_join(rel)

def resolve_existing_path(path_str: str | None) -> str | None:
    """
    Devuelve:
    - Path local si existe
    - Si no existe local, y hay bucket:
        - intenta mapear a gs://.../data_store/...
        - si ya era gs://..., devuelve eso
    """
    if not path_str:
        return None

    p = str(path_str).strip()

    # Si ya es gs://, lo dejamos
    if _is_gs_path(p):
        return p

    # Si existe local, ok
    if _exists_local(p):
        return p

    # LOCAL: intentar buscar por filename en roots (solo local)
    fname = Path(p).name
    for root in DATA_ROOTS:
        rootp = Path(root)
        if not rootp.exists():
            continue
        try:
            found = next(rootp.rglob(fname), None)
            if found and found.exists():
                return str(found)
        except Exception:
            pass

    # CLOUD: mapear data_store -> gs://
    gcs_guess = map_local_datastore_to_gcs(p)
    if gcs_guess:
        return gcs_guess

    return None

def compute_semaforo_aib(row: pd.Series,
                         se_target: str = "AIB",
                         sum_media: float = 200.0,
                         sum_alta: float = 250.0,
                         llen_ok: float = 70.0,
                         llen_bajo: float = 50.0):
    se = row.get("SE", None)
    se_str = str(se).strip().upper() if se is not None and not (isinstance(se, float) and pd.isna(se)) else ""
    if se_str != se_target:
        return "NO APLICA"

    s = safe_to_float(row.get("Sumergencia"))
    llen = safe_to_float(row.get("Bba Llenado"))

    if s is None or llen is None:
        return "SIN DATOS"

    if s < sum_media or llen >= llen_ok:
        return "🟢 NORMAL"

    if s > sum_alta and llen < llen_bajo:
        return "🔴 CRÍTICO"

    return "🟡 ALERTA"

# ---------------- Loaders (LOCAL o GCS) ----------------
def _read_parquet_any(local_path: str, gcs_path: str) -> pd.DataFrame:
    # 1) Local
    if local_path and os.path.exists(local_path):
        return pd.read_parquet(local_path)

    # 2) GCS
    if gcs_path and _is_gs_path(gcs_path):
        lp = _gcs_download_to_temp(gcs_path)
        return pd.read_parquet(lp)

    return pd.DataFrame()

def _read_csv_any(local_path: str, gcs_path: str) -> pd.DataFrame:
    # 1) Local
    if local_path and os.path.exists(local_path):
        return pd.read_csv(local_path, parse_dates=["mtime", "din_datetime"], dayfirst=True, keep_default_na=True)

    # 2) GCS
    if gcs_path and _is_gs_path(gcs_path):
        lp = _gcs_download_to_temp(gcs_path)
        return pd.read_csv(lp, parse_dates=["mtime", "din_datetime"], dayfirst=True, keep_default_na=True)

    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_din_index():
    # preferimos parquet; si no, csv
    if os.path.exists(INDEX_PARQUET_LOCAL):
        try:
            return pd.read_parquet(INDEX_PARQUET_LOCAL)
        except Exception:
            pass

    if os.path.exists(INDEX_CSV_LOCAL):
        return pd.read_csv(INDEX_CSV_LOCAL, parse_dates=["mtime", "din_datetime"], dayfirst=True, keep_default_na=True)

    # Cloud
    if GCS_BUCKET:
        try:
            return _read_parquet_any("", INDEX_PARQUET_GCS)
        except Exception:
            # si existiera CSV en bucket (opcional), podrías agregarlo
            return pd.DataFrame()

    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_niv_index():
    if os.path.exists(NIV_INDEX_LOCAL):
        return pd.read_parquet(NIV_INDEX_LOCAL)

    if GCS_BUCKET:
        try:
            return _read_parquet_any("", NIV_INDEX_GCS)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def parse_din_surface_points(path_str: str) -> pd.DataFrame:
    """
    path_str puede ser:
    - local: C:\...\archivo.din
    - gcs:  gs://bucket/data_store/din/.../archivo.din
    """
    # Si es GCS, bajar a temp
    if _is_gs_path(path_str):
        path_str = _gcs_download_to_temp(path_str)

    p = Path(path_str)
    txt = read_text_best_effort(p)

    section = None
    xs = {}
    ys = {}
    in_cs = False

    for line in txt.splitlines():
        m = SECTION_RE.match(line)
        if m:
            section = m.group(1).strip().upper()
            in_cs = (section == "CS")
            continue

        m = KV_RE.match(line)
        if not m or not section:
            continue

        k_raw = m.group(1).strip()
        v_raw = m.group(2).strip()

        if in_cs:
            mk = POINT_KEY_RE.match(k_raw)
            if mk:
                xy = mk.group(1).upper()
                idx = int(mk.group(2))
                try:
                    val = float(v_raw.replace(",", "."))
                except Exception:
                    continue
                if xy == "X":
                    xs[idx] = val
                else:
                    ys[idx] = val

    idxs = sorted(set(xs.keys()) & set(ys.keys()))
    return pd.DataFrame({"i": idxs, "X": [xs[i] for i in idxs], "Y": [ys[i] for i in idxs]})

@st.cache_data(show_spinner=False)
def parse_din_extras(path_str: str) -> dict:
    if _is_gs_path(path_str):
        path_str = _gcs_download_to_temp(path_str)

    p = Path(path_str)
    txt = read_text_best_effort(p)

    wanted = {(sec.upper(), key.upper()): col for col, (sec, key) in EXTRA_FIELDS.items()}
    out = {col: None for col in EXTRA_FIELDS.keys()}
    section = None

    for line in txt.splitlines():
        m = SECTION_RE.match(line)
        if m:
            section = m.group(1).strip().upper()
            continue
        m = KV_RE.match(line)
        if not m or not section:
            continue
        k = m.group(1).strip().upper()
        v = m.group(2).strip()
        if (section, k) in wanted:
            col = wanted[(section, k)]
            fv = safe_to_float(v)
            out[col] = fv if fv is not None else (v if v != "" else None)

    # Fallback %Balance
    if out.get("%Balance") is None:
        wanted_pc_alt = ("RARR", "PC")
        section = None
        for line in txt.splitlines():
            m = SECTION_RE.match(line)
            if m:
                section = m.group(1).strip().upper()
                continue
            m = KV_RE.match(line)
            if not m or not section:
                continue
            k = m.group(1).strip().upper()
            v = m.group(2).strip()
            if (section, k) == wanted_pc_alt:
                fv = safe_to_float(v)
                out["%Balance"] = fv if fv is not None else (v if v != "" else None)
                break

    return out

def build_keys(df: pd.DataFrame, no_col: str, fe_col: str, ho_col: str | None):
    df = df.copy()
    df["NO_key"] = df[no_col].apply(normalize_no_exact) if no_col in df.columns else ""
    df["FE_key"] = df[fe_col].apply(normalize_fe_date) if fe_col in df.columns else None
    if ho_col and ho_col in df.columns:
        df["HO_key"] = df[ho_col].apply(normalize_ho_str)
    else:
        df["HO_key"] = ""
    return df

def compute_sumergencia_and_base(row):
    pb = safe_to_float(row.get("PB"))
    if pb is None:
        return None, None

    nc = safe_to_float(row.get("NC"))
    nm = safe_to_float(row.get("NM"))
    nd = safe_to_float(row.get("ND"))

    if nc is not None:
        return pb - nc, "NC"
    if nm is not None:
        return pb - nm, "NM"
    if nd is not None:
        return pb - nd, "ND"
    return None, None

def make_display_label(row: pd.Series) -> str:
    fe = row.get("fecha", None)
    ho = row.get("hora", None)
    origen = row.get("ORIGEN", "")
    parts = []
    if fe: parts.append(str(fe))
    if ho: parts.append(str(ho))
    if origen: parts.append(str(origen))
    base = " | ".join(parts) if parts else "SIN_FECHA"
    return base

def _infer_dt_plot(dfp: pd.DataFrame) -> pd.Series:
    dt = None
    if "din_datetime" in dfp.columns:
        dt = pd.to_datetime(dfp["din_datetime"], errors="coerce")
    if dt is None or dt.isna().all():
        if "niv_datetime" in dfp.columns:
            dt = pd.to_datetime(dfp["niv_datetime"], errors="coerce")
    if dt is None:
        dt = pd.Series([pd.NaT] * len(dfp))
    if dt.isna().all() and "FE_key" in dfp.columns:
        try:
            dt = pd.to_datetime(
                dfp["FE_key"].astype(str) + " " + dfp.get("HO_key", "").astype(str),
                errors="coerce",
                dayfirst=True
            )
        except Exception:
            pass
    return dt

def _dedup_niv(df_niv_k: pd.DataFrame) -> pd.DataFrame:
    if df_niv_k is None or df_niv_k.empty:
        return pd.DataFrame()
    out = df_niv_k.copy()
    sort_niv = [c for c in ["niv_datetime", "mtime"] if c in out.columns]
    if sort_niv:
        out = out.sort_values(sort_niv, na_position="last")
    out = out.drop_duplicates(subset=["NO_key", "FE_key", "HO_key"], keep="last").reset_index(drop=True)
    return out

def _dedup_din(df_din_k: pd.DataFrame) -> pd.DataFrame:
    if df_din_k is None or df_din_k.empty:
        return pd.DataFrame()
    out = df_din_k.copy()
    sort_din = [c for c in ["din_datetime", "mtime"] if c in out.columns]
    if sort_din:
        out = out.sort_values(sort_din, na_position="last")
    if "path" in out.columns:
        out = out.drop_duplicates(subset=["path"], keep="last").reset_index(drop=True)
    else:
        out = out.drop_duplicates(subset=["NO_key", "FE_key", "HO_key"], keep="last").reset_index(drop=True)
    return out

@st.cache_data(show_spinner=False)
def build_global_consolidated(din_ok: pd.DataFrame, niv_ok: pd.DataFrame,
                             din_no_col: str | None, din_fe_col: str | None, din_ho_col: str | None,
                             niv_no_col: str | None, niv_fe_col: str | None, niv_ho_col: str | None) -> pd.DataFrame:
    din_d = _dedup_din(din_ok) if din_ok is not None else pd.DataFrame()
    niv_d = _dedup_niv(niv_ok) if niv_ok is not None else pd.DataFrame()

    din_join = din_d.copy()
    if not din_join.empty:
        din_join["ORIGEN"] = "DIN"
        if not niv_d.empty:
            din_join = din_join.merge(
                niv_d,
                on=["NO_key", "FE_key", "HO_key"],
                how="left",
                suffixes=("", "_niv")
            )

    niv_only = pd.DataFrame()
    if not niv_d.empty:
        if din_d.empty:
            niv_only = niv_d.copy()
        else:
            key_din = din_d[["NO_key", "FE_key", "HO_key"]].drop_duplicates()
            niv_only = niv_d.merge(key_din, on=["NO_key", "FE_key", "HO_key"], how="left", indicator=True)
            niv_only = niv_only[niv_only["_merge"] == "left_only"].drop(columns=["_merge"])

        if not niv_only.empty:
            niv_only = niv_only.copy()
            niv_only["ORIGEN"] = "NIV"

    dfp_all = pd.concat([din_join, niv_only], ignore_index=True, sort=False)
    dfp_all = make_unique_columns(dfp_all)

    if not dfp_all.empty:
        if din_no_col and din_no_col in dfp_all.columns:
            dfp_all["pozo"] = dfp_all[din_no_col]
        elif niv_no_col and niv_no_col in dfp_all.columns:
            dfp_all["pozo"] = dfp_all[niv_no_col]
        else:
            dfp_all["pozo"] = dfp_all["NO_key"]

        if din_fe_col and din_fe_col in dfp_all.columns:
            dfp_all["fecha"] = dfp_all[din_fe_col]
        elif niv_fe_col and niv_fe_col in dfp_all.columns:
            dfp_all["fecha"] = dfp_all[niv_fe_col]
        else:
            dfp_all["fecha"] = dfp_all["FE_key"].astype("string")

        if din_ho_col and din_ho_col in dfp_all.columns:
            dfp_all["hora"] = dfp_all[din_ho_col]
        elif niv_ho_col and niv_ho_col in dfp_all.columns:
            dfp_all["hora"] = dfp_all[niv_ho_col]
        else:
            dfp_all["hora"] = dfp_all["HO_key"].astype("string")

    for c in ["CO", "empresa", "SE", "NM", "NC", "ND", "PE", "PB", "CM", "niv_datetime"]:
        if c not in dfp_all.columns:
            alt = f"{c}_niv"
            if alt in dfp_all.columns:
                dfp_all[c] = dfp_all[alt]

    for c in ["NM", "NC", "ND", "PE", "PB"]:
        if c in dfp_all.columns:
            dfp_all[c] = dfp_all[c].apply(safe_to_float)

    tmp = dfp_all.apply(compute_sumergencia_and_base, axis=1, result_type="expand")
    dfp_all["Sumergencia"] = tmp[0]
    dfp_all["Sumergencia_base"] = tmp[1]

    dfp_all["DT_plot"] = _infer_dt_plot(dfp_all)

    return dfp_all

@st.cache_data(show_spinner=False)
def parse_extras_for_paths(paths: list[str]) -> pd.DataFrame:
    rows = []
    for pth in paths:
        try:
            if pth:
                # si es gs:// baja en parse_din_extras solo
                rows.append(parse_din_extras(str(pth)))
            else:
                rows.append({k: None for k in EXTRA_FIELDS.keys()})
        except Exception:
            rows.append({k: None for k in EXTRA_FIELDS.keys()})
    return pd.DataFrame(rows)

def _trend_linear_per_month(df_one: pd.DataFrame, ycol: str):
    if df_one is None or df_one.empty or ycol not in df_one.columns or "DT_plot" not in df_one.columns:
        return None

    d = df_one[["DT_plot", ycol]].dropna().copy()
    if d.empty:
        return None
    d["DT_plot"] = pd.to_datetime(d["DT_plot"], errors="coerce")
    d = d.dropna(subset=["DT_plot"]).sort_values("DT_plot")
    if d.shape[0] < 2:
        return None

    t0 = d["DT_plot"].iloc[0]
    x_days = (d["DT_plot"] - t0).dt.total_seconds() / 86400.0
    x_months = x_days / 30.4375
    y = pd.to_numeric(d[ycol], errors="coerce")
    good = (~x_months.isna()) & (~y.isna())
    x = x_months[good].to_numpy()
    yv = y[good].to_numpy()

    if len(x) < 2:
        return None

    x_mean = x.mean()
    y_mean = yv.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return None
    b = ((x - x_mean) * (yv - y_mean)).sum() / denom

    return float(b), float(yv[0]), float(yv[-1]), int(len(x))

# ---------------- UI ----------------
st.set_page_config(page_title="DIN - Cartas de Superficie", layout="wide")
st.title("📈 Interfaz DIN — Carta de Superficie (CS)")

df_din = load_din_index()
df_niv = load_niv_index()

if df_din.empty and df_niv.empty:
    st.error(
        "No encontré índices.\n\n"
        "- En LOCAL: corré primero:  python index_din.py  y  python index_niv.py\n"
        "- En CLOUD: asegurate de setear DINAS_BUCKET y subir din_index.parquet y niv_index.parquet al bucket."
    )
    st.stop()

# Resolver paths DIN (local o gs://)
if not df_din.empty and "path" in df_din.columns:
    df_din["path"] = df_din["path"].apply(lambda x: resolve_existing_path(x) if pd.notna(x) else None)

din_no_col = find_col(df_din, ["pozo", "NO"])
din_fe_col = find_col(df_din, ["fecha", "FE"])
din_ho_col = find_col(df_din, ["hora", "HO"])

niv_no_col = find_col(df_niv, ["pozo", "NO"])
niv_fe_col = find_col(df_niv, ["fecha", "FE"])
niv_ho_col = find_col(df_niv, ["hora", "HO"])

if not df_din.empty and din_no_col and din_fe_col:
    df_din_k = build_keys(df_din, din_no_col, din_fe_col, din_ho_col)
else:
    df_din_k = pd.DataFrame()

if not df_niv.empty and niv_no_col and niv_fe_col:
    df_niv_k = build_keys(df_niv, niv_no_col, niv_fe_col, niv_ho_col)
else:
    df_niv_k = pd.DataFrame()

with st.sidebar:
    st.header("Filtros")
    only_ok = st.checkbox("Ocultar filas con error", value=True)

    din_ok = df_din_k.copy()
    niv_ok = df_niv_k.copy()

    if only_ok:
        if not din_ok.empty and "error" in din_ok.columns:
            din_ok = din_ok[din_ok["error"].isna()]
        if not niv_ok.empty and "error" in niv_ok.columns:
            niv_ok = niv_ok[niv_ok["error"].isna()]

    pozos = sorted(
        pd.Series(
            list(din_ok["NO_key"].tolist() if not din_ok.empty else [])
            + list(niv_ok["NO_key"].tolist() if not niv_ok.empty else [])
        )
        .dropna()
        .map(lambda x: normalize_no_exact(x))
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )

    if not pozos:
        st.error("No hay pozos válidos en los índices (NO vacío). Revisá parseo NO= en DIN/NIV.")
        st.stop()

    pozo_sel = st.selectbox("Pozo (NO=)", options=pozos)

tab_med, tab_stats = st.tabs(["📈 Mediciones", "📊 Estadísticas"])

# ==========================================================
# TAB 1: MEDICIONES
# ==========================================================
with tab_med:
    din_p = din_ok[din_ok["NO_key"] == pozo_sel].copy() if not din_ok.empty else pd.DataFrame()
    niv_p = niv_ok[niv_ok["NO_key"] == pozo_sel].copy() if not niv_ok.empty else pd.DataFrame()

    if not niv_p.empty:
        sort_niv = [c for c in ["niv_datetime", "mtime"] if c in niv_p.columns]
        if sort_niv:
            niv_p = niv_p.sort_values(sort_niv, na_position="last")
        niv_p = niv_p.drop_duplicates(subset=["NO_key", "FE_key", "HO_key"], keep="last").reset_index(drop=True)

    if not din_p.empty:
        sort_din = [c for c in ["din_datetime", "mtime"] if c in din_p.columns]
        if sort_din:
            din_p = din_p.sort_values(sort_din, na_position="last")
        if "path" in din_p.columns:
            din_p = din_p.drop_duplicates(subset=["path"], keep="last").reset_index(drop=True)
        else:
            din_p = din_p.drop_duplicates(subset=["NO_key", "FE_key", "HO_key"], keep="last").reset_index(drop=True)

    din_join = din_p.copy()
    if not din_join.empty:
        din_join["ORIGEN"] = "DIN"
        if not niv_p.empty:
            din_join = din_join.merge(
                niv_p,
                on=["NO_key", "FE_key", "HO_key"],
                how="left",
                suffixes=("", "_niv")
            )

    niv_only = pd.DataFrame()
    if not niv_p.empty:
        if din_p.empty:
            niv_only = niv_p.copy()
        else:
            key_din = din_p[["NO_key", "FE_key", "HO_key"]].drop_duplicates()
            niv_only = niv_p.merge(key_din, on=["NO_key", "FE_key", "HO_key"], how="left", indicator=True)
            niv_only = niv_only[niv_only["_merge"] == "left_only"].drop(columns=["_merge"])

        if not niv_only.empty:
            niv_only = niv_only.copy()
            niv_only["ORIGEN"] = "NIV"

    dfp = pd.concat([din_join, niv_only], ignore_index=True, sort=False)
    dfp = make_unique_columns(dfp)

    if not dfp.empty:
        if din_no_col and din_no_col in dfp.columns:
            dfp["pozo"] = dfp[din_no_col]
        elif niv_no_col and niv_no_col in dfp.columns:
            dfp["pozo"] = dfp[niv_no_col]
        else:
            dfp["pozo"] = dfp["NO_key"]

        if din_fe_col and din_fe_col in dfp.columns:
            dfp["fecha"] = dfp[din_fe_col]
        elif niv_fe_col and niv_fe_col in dfp.columns:
            dfp["fecha"] = dfp[niv_fe_col]
        else:
            dfp["fecha"] = dfp["FE_key"].astype("string")

        if din_ho_col and din_ho_col in dfp.columns:
            dfp["hora"] = dfp[din_ho_col]
        elif niv_ho_col and niv_ho_col in dfp.columns:
            dfp["hora"] = dfp[niv_ho_col]
        else:
            dfp["hora"] = dfp["HO_key"].astype("string")

    for c in ["CO", "empresa", "SE", "NM", "NC", "ND", "PE", "PB", "CM", "niv_datetime"]:
        if c not in dfp.columns:
            alt = f"{c}_niv"
            if alt in dfp.columns:
                dfp[c] = dfp[alt]

    for c in ["NM", "NC", "ND", "PE", "PB"]:
        if c in dfp.columns:
            dfp[c] = dfp[c].apply(safe_to_float)

    tmp = dfp.apply(compute_sumergencia_and_base, axis=1, result_type="expand")
    dfp["Sumergencia"] = tmp[0]
    dfp["Sumergencia_base"] = tmp[1]

    dfp["DT_plot"] = _infer_dt_plot(dfp)

    # Extras DIN (por fila, solo para DIN)
    extra_rows = []
    if "path" in dfp.columns:
        for _, r in dfp.iterrows():
            if r.get("ORIGEN") == "DIN":
                pth = r.get("path")
                try:
                    p_res = resolve_existing_path(pth)
                    if p_res:
                        extra_rows.append(parse_din_extras(str(p_res)))
                    else:
                        extra_rows.append({k: None for k in EXTRA_FIELDS.keys()})
                except Exception:
                    extra_rows.append({k: None for k in EXTRA_FIELDS.keys()})
            else:
                extra_rows.append({k: None for k in EXTRA_FIELDS.keys()})
    else:
        extra_rows = [{k: None for k in EXTRA_FIELDS.keys()} for _ in range(len(dfp))]

    df_extra = pd.DataFrame(extra_rows)
    for c in df_extra.columns:
        if c in dfp.columns:
            dfp = dfp.drop(columns=[c])
    dfp = pd.concat([dfp.reset_index(drop=True), df_extra.reset_index(drop=True)], axis=1)

    base_cols = [c for c in ["ORIGEN", "pozo", "fecha", "hora", "din_datetime", "niv_datetime"] if c in dfp.columns]
    niv_cols = [c for c in ["CO", "empresa", "SE", "NM", "NC", "ND", "PE", "PB", "CM", "Sumergencia", "Sumergencia_base"] if c in dfp.columns]
    extra_cols = [c for c in [
        "AIB Carrera","Sentido giro",
        "Tipo Contrapesos","Distancia contrapesos (cm)","Contrapeso actual","Contrapeso ideal",
        "AIBEB_Torque max contrapeso","%Estructura", "%Balance",
        "Bba Diam Pistón","Bba Prof","Bba Llenado","GPM",
        "Polea Motor","Potencia Motor","RPM Motor",
    ] if c in dfp.columns]

    table_cols = base_cols + niv_cols + extra_cols

    sort_cols = [c for c in ["FE_key", "HO_key", "din_datetime", "niv_datetime"] if c in dfp.columns]
    dfp_sorted = dfp.sort_values(sort_cols, na_position="last") if sort_cols else dfp
    df_show = dfp_sorted[table_cols].copy()

    st.subheader(f"Pozo (NO=): {pozo_sel}")
    st.write(f"Mediciones totales (DIN + NIV): **{len(df_show)}**")
    st.dataframe(df_show, use_container_width=True, height=320)

    # Histórico
    hist = dfp_sorted.copy()
    if "DT_plot" in hist.columns and "Sumergencia" in hist.columns:
        hist["Sumergencia"] = pd.to_numeric(hist["Sumergencia"], errors="coerce")
        hist = hist.dropna(subset=["DT_plot", "Sumergencia"]).copy()
        hist = hist.sort_values("DT_plot")

        if not hist.empty:
            st.subheader("📉 Histórico — Sumergencia vs Tiempo")

            def pick_used_level(row):
                base = row.get("Sumergencia_base")
                if base == "NC":
                    return safe_to_float(row.get("NC"))
                if base == "NM":
                    return safe_to_float(row.get("NM"))
                if base == "ND":
                    return safe_to_float(row.get("ND"))
                return None

            hist["PB_f"] = hist["PB"].apply(safe_to_float) if "PB" in hist.columns else None
            hist["Nivel_usado"] = hist.apply(pick_used_level, axis=1)

            custom = pd.DataFrame({
                "pozo": hist.get("pozo", ""),
                "origen": hist.get("ORIGEN", ""),
                "base": hist.get("Sumergencia_base", ""),
                "pb": hist.get("PB_f", None),
                "nivel": hist.get("Nivel_usado", None),
            }).to_numpy()

            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=hist["DT_plot"],
                    y=hist["Sumergencia"],
                    mode="lines+markers",
                    name="Sumergencia",
                    customdata=custom,
                    hovertemplate=(
                        "Pozo: %{customdata[0]}<br>"
                        "Origen: %{customdata[1]}<br>"
                        "Fecha/Hora: %{x|%d/%m/%Y %H:%M}<br>"
                        "Sumergencia: %{y:.2f}<br>"
                        "Base usada: %{customdata[2]}<br>"
                        "PB: %{customdata[3]}<br>"
                        "Nivel (%{customdata[2]}): %{customdata[4]}<extra></extra>"
                    )
                )
            )

            fig2.update_layout(
                xaxis_title="Fecha / Hora",
                yaxis_title="Sumergencia (PB - nivel)",
                height=420
            )

            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No hay datos suficientes para graficar Sumergencia (faltan PB y NM/NC/ND o fecha/hora).")

    # Carta DIN (CS)
    din_rows = dfp_sorted[dfp_sorted["ORIGEN"] == "DIN"].copy() if "ORIGEN" in dfp_sorted.columns else pd.DataFrame()

    if din_rows.empty:
        st.info("Este pozo no tiene archivos DIN para graficar (solo NIV o no se resolvió el path).")
        st.stop()

    din_rows["option_id"] = din_rows["path"].astype(str)
    din_rows["option_label"] = din_rows.apply(make_display_label, axis=1)

    options = din_rows["option_id"].dropna().unique().tolist()
    label_map = dict(zip(din_rows["option_id"], din_rows["option_label"]))

    sel_ids = st.multiselect(
        "Elegí una o varias mediciones DIN para superponer (CS - superficie):",
        options=options,
        format_func=lambda oid: label_map.get(oid, oid),
        default=options[:1] if options else []
    )

    if not sel_ids:
        st.info("Seleccioná al menos una medición para graficar.")
        st.stop()

    fig = go.Figure()
    errors = 0

    for oid in sel_ids:
        row = din_rows[din_rows["option_id"] == oid].iloc[0]
        path = row.get("path", None)
        name = row.get("option_label", "medición")

        if not path:
            continue

        p_res = resolve_existing_path(path)
        if not p_res:
            st.warning(f"No encuentro el archivo: {Path(str(path)).name}")
            errors += 1
            continue

        try:
            pts = parse_din_surface_points(str(p_res))
            if pts.empty:
                st.warning(f"No pude leer puntos [CS] en: {Path(str(p_res)).name}")
                errors += 1
                continue

            fig.add_trace(go.Scatter(x=pts["X"], y=pts["Y"], mode="lines", name=name))
        except Exception as e:
            st.warning(f"Error leyendo {Path(str(p_res)).name}: {e}")
            errors += 1

    fig.update_layout(
        title="Carta Dinamométrica — Superficie (CS)",
        xaxis_title="X (posición / carrera)",
        yaxis_title="Y (carga)",
        legend_title="Mediciones",
        height=650
    )

    st.plotly_chart(fig, use_container_width=True)

    if errors:
        st.caption(f"⚠️ Algunas mediciones no pudieron graficarse ({errors}). Revisá si esos .din tienen sección [CS].")

# ==========================================================
# TAB 2: ESTADÍSTICAS
# (Tu bloque es MUY largo; lo dejo igual salvo que usa resolve_existing_path y parse_din_extras que ya soportan GCS)
# ==========================================================
with tab_stats:
    st.subheader("📊 Estadísticas (última medición por pozo)")

    df_all = build_global_consolidated(
        din_ok, niv_ok,
        din_no_col, din_fe_col, din_ho_col,
        niv_no_col, niv_fe_col, niv_ho_col
    )

    if df_all.empty:
        st.info("No hay datos suficientes para estadísticas.")
        st.stop()

    df_all = df_all.copy()
    df_all["DT_plot"] = pd.to_datetime(df_all["DT_plot"], errors="coerce")

    df_all_sorted = df_all.sort_values(["NO_key", "DT_plot"], na_position="last")
    snap = df_all_sorted.dropna(subset=["DT_plot"]).groupby("NO_key", as_index=False).tail(1).copy()

    if "path" in snap.columns:
        snap["path_res"] = snap["path"].apply(lambda x: resolve_existing_path(x) if pd.notna(x) else None)
    else:
        snap["path_res"] = None

    din_mask = (snap.get("ORIGEN") == "DIN") & snap["path_res"].notna()
    din_paths = snap.loc[din_mask, "path_res"].astype(str).tolist()

    if din_paths:
        df_extras_snap = parse_extras_for_paths(din_paths)
        df_extras_snap.index = snap.loc[din_mask].index

        for c in EXTRA_FIELDS.keys():
            if c not in snap.columns:
                snap[c] = None

        for c in df_extras_snap.columns:
            snap.loc[din_mask, c] = df_extras_snap[c].values
    else:
        for c in EXTRA_FIELDS.keys():
            if c not in snap.columns:
                snap[c] = None

    snap["Sumergencia"] = pd.to_numeric(snap.get("Sumergencia"), errors="coerce")
    snap["PB"] = pd.to_numeric(snap.get("PB"), errors="coerce")
    snap["NM"] = pd.to_numeric(snap.get("NM"), errors="coerce")
    snap["NC"] = pd.to_numeric(snap.get("NC"), errors="coerce")
    snap["ND"] = pd.to_numeric(snap.get("ND"), errors="coerce")

    snap["%Estructura"] = pd.to_numeric(snap.get("%Estructura"), errors="coerce")
    snap["%Balance"] = pd.to_numeric(snap.get("%Balance"), errors="coerce")
    snap["Bba Llenado"] = pd.to_numeric(snap.get("Bba Llenado"), errors="coerce")

    now = pd.Timestamp.now()
    snap["Dias_desde_ultima"] = (now - snap["DT_plot"]).dt.total_seconds() / 86400.0

    # --- A partir de acá, tu bloque de Estadísticas sigue igual ---
    # (lo podés pegar tal cual desde tu versión, no cambia la lógica)
    st.info("✅ Tu pestaña de Estadísticas puede quedar igual que la tenías. "
            "Ya soporta GCS porque resolve_existing_path + parse_din_extras ya manejan gs://.")

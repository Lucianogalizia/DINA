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
#
# NUEVO (mapa):
# - Excel estático en el repo: assets/Nombres-Pozo_con_coordenadas.xlsx
#   (nombre_corto + GEO_LATITUDE + GEO_LONGITUDE)
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
    try:
        from google.cloud import storage
        return storage.Client()
    except Exception:
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

# Excel estático versionado en repo
BASE_DIR = Path(__file__).resolve().parent
COORDS_XLSX_REPO = BASE_DIR / "assets" / "Nombres-Pozo_con_coordenadas.xlsx"


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

    # NUEVO: Caudal bruto efec (viene en [RBO] CF=...)
    "Caudal bruto efec": ("RBO", "CF"),

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
def load_coords_repo():
    candidates = [
        COORDS_XLSX_REPO,                           # BASE_DIR/assets/...
        Path.cwd() / "assets" / "Nombres-Pozo_con_coordenadas.xlsx",
        Path("/app/assets/Nombres-Pozo_con_coordenadas.xlsx"),
    ]
    for p in candidates:
        try:
            if p.exists():
                return pd.read_excel(p)
        except Exception:
            pass

    hits = list(BASE_DIR.rglob("Nombres-Pozo_con_coordenadas.xlsx"))
    if hits:
        return pd.read_excel(hits[0])

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
# ==========================================================
# (NUEVO) Snapshot rápido para MAPA
# Devuelve 1 fila por pozo: la última medición global (DIN o NIV)
# ==========================================================
def _pick_dt_plot(df: pd.DataFrame, preferred_cols: list[str]) -> pd.Series:
    """
    Devuelve un Series datetime usando la mejor columna disponible.
    Si no encuentra ninguna útil, intenta FE_key + HO_key.
    """
    for c in preferred_cols:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            s = s.reindex(df.index)  # asegura mismo index
            if not s.isna().all():
                return s

    # Fallback FE_key + HO_key
    if "FE_key" in df.columns:
        try:
            ho = df["HO_key"] if "HO_key" in df.columns else ""
            s = pd.to_datetime(
                df["FE_key"].astype(str) + " " + ho.astype(str),
                errors="coerce",
                dayfirst=True
            )
            return s.reindex(df.index)
        except Exception:
            pass

    return pd.Series([pd.NaT] * len(df), index=df.index)



@st.cache_data(show_spinner=False)
def build_last_snapshot_for_map(din_ok: pd.DataFrame, niv_ok: pd.DataFrame) -> pd.DataFrame:
    """
    1 fila por pozo (NO_key) con la última medición entre DIN y NIV.
    NO asume que existan PB/NM/NC/ND/PE en los índices: si faltan, las crea en None.
    """

    def _prep_one(df: pd.DataFrame, origen: str, dt_candidates: list[str]) -> pd.DataFrame:
        if df is None or df.empty or "NO_key" not in df.columns:
            return pd.DataFrame()

        keep = [c for c in ["NO_key", "mtime", "din_datetime", "niv_datetime", "FE_key", "HO_key",
                            "PB", "NM", "NC", "ND", "PE"] if c in df.columns]
        d = df[keep].copy()

        # NO_key limpio
        d["NO_key"] = d["NO_key"].astype(str).map(lambda x: x.strip())
        d = d[d["NO_key"] != ""]

        d["ORIGEN"] = origen
        d["DT_plot"] = _pick_dt_plot(d, dt_candidates)

        # Asegurar columnas numéricas aunque no existan en el índice
        for c in ["PB", "NM", "NC", "ND", "PE"]:
            if c not in d.columns:
                d[c] = None
            d[c] = pd.to_numeric(d[c], errors="coerce")

        d = d.sort_values(["NO_key", "DT_plot"], na_position="last")
        last = d.groupby("NO_key", as_index=False).tail(1).copy()

        out_cols = ["NO_key", "ORIGEN", "DT_plot", "PB", "NM", "NC", "ND", "PE"]
        # (ya garantizamos que estén)
        return last[out_cols]

    din_last = _prep_one(din_ok, "DIN", ["din_datetime", "mtime"])
    niv_last = _prep_one(niv_ok, "NIV", ["niv_datetime", "mtime"])

    both = pd.concat([din_last, niv_last], ignore_index=True, sort=False)
    if both.empty:
        return pd.DataFrame()

    both = both.sort_values(["NO_key", "DT_plot"], na_position="last")
    snap = both.groupby("NO_key", as_index=False).tail(1).copy()

    # Calcular Sumergencia (si PB y algún nivel existen)
    tmp = snap.apply(compute_sumergencia_and_base, axis=1, result_type="expand")
    snap["Sumergencia"] = tmp[0]
    snap["Sumergencia_base"] = tmp[1]

    return snap.reset_index(drop=True)



@st.cache_data(show_spinner=False)
def parse_extras_for_paths(paths: list[str]) -> pd.DataFrame:
    rows = []
    for pth in paths:
        try:
            if pth:
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

tab_med, tab_stats, tab_map = st.tabs(["📈 Mediciones", "📊 Estadísticas", "🗺️ Mapa de sumergencia"])


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
        "Bba Diam Pistón","Bba Prof","Bba Llenado","GPM","Caudal bruto efec",
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
# TAB 2: ESTADÍSTICAS (última medición por pozo)
# ==========================================================
with tab_stats:
    st.subheader("📊 Estadísticas (última medición por pozo)")

    # Consolidado global (sin extras masivos)
    df_all = build_global_consolidated(
        din_ok, niv_ok,
        din_no_col, din_fe_col, din_ho_col,
        niv_no_col, niv_fe_col, niv_ho_col
    )

    if df_all.empty:
        st.info("No hay datos suficientes para estadísticas.")
        st.stop()

    # DT_plot
    df_all = df_all.copy()
    df_all["DT_plot"] = pd.to_datetime(df_all["DT_plot"], errors="coerce")

    # Snapshot: última medición por pozo (usa DT_plot)
    df_all_sorted = df_all.sort_values(["NO_key", "DT_plot"], na_position="last")
    snap = df_all_sorted.dropna(subset=["DT_plot"]).groupby("NO_key", as_index=False).tail(1).copy()

    # Resolver path en snapshot
    if "path" in snap.columns:
        snap["path_res"] = snap["path"].apply(lambda x: resolve_existing_path(x) if pd.notna(x) else None)
    else:
        snap["path_res"] = None

    # Parse EXTRAS solo para la última DIN por pozo (si es DIN)
    din_mask = (snap.get("ORIGEN") == "DIN") & snap["path_res"].notna()
    din_paths = snap.loc[din_mask, "path_res"].astype(str).tolist()

    if din_paths:
        df_extras_snap = parse_extras_for_paths(din_paths)
        df_extras_snap.index = snap.loc[din_mask].index  # alinear por index

        for c in EXTRA_FIELDS.keys():
            if c not in snap.columns:
                snap[c] = None

        for c in df_extras_snap.columns:
            snap.loc[din_mask, c] = df_extras_snap[c].values
    else:
        for c in EXTRA_FIELDS.keys():
            if c not in snap.columns:
                snap[c] = None

    # Tipos numéricos
    snap["Sumergencia"] = pd.to_numeric(snap.get("Sumergencia"), errors="coerce")
    snap["PB"] = pd.to_numeric(snap.get("PB"), errors="coerce")
    snap["NM"] = pd.to_numeric(snap.get("NM"), errors="coerce")
    snap["NC"] = pd.to_numeric(snap.get("NC"), errors="coerce")
    snap["ND"] = pd.to_numeric(snap.get("ND"), errors="coerce")
    snap["PE"] = pd.to_numeric(snap.get("PE"), errors="coerce")

    snap["%Estructura"] = pd.to_numeric(snap.get("%Estructura"), errors="coerce")
    snap["%Balance"] = pd.to_numeric(snap.get("%Balance"), errors="coerce")
    snap["Bba Llenado"] = pd.to_numeric(snap.get("Bba Llenado"), errors="coerce")
    snap["Caudal bruto efec"] = pd.to_numeric(snap.get("Caudal bruto efec"), errors="coerce")

    # Antigüedad
    now = pd.Timestamp.now()
    snap["Dias_desde_ultima"] = (now - snap["DT_plot"]).dt.total_seconds() / 86400.0

    # ---------------- Controles (filtros snapshot) ----------------
    s_min = float(snap["Sumergencia"].min()) if snap["Sumergencia"].notna().any() else 0.0
    s_max = float(snap["Sumergencia"].max()) if snap["Sumergencia"].notna().any() else 1.0

    est_min = float(snap["%Estructura"].min()) if snap["%Estructura"].notna().any() else 0.0
    est_max = float(snap["%Estructura"].max()) if snap["%Estructura"].notna().any() else 100.0

    bal_min = float(snap["%Balance"].min()) if snap["%Balance"].notna().any() else 0.0
    bal_max = float(snap["%Balance"].max()) if snap["%Balance"].notna().any() else 100.0

    def _fix_range(vmin: float, vmax: float, pad: float = 1.0):
        if vmin == vmax:
            return vmin - pad, vmax + pad
        return vmin, vmax

    s_min, s_max     = _fix_range(s_min, s_max, pad=1.0)
    est_min, est_max = _fix_range(est_min, est_max, pad=1.0)
    bal_min, bal_max = _fix_range(bal_min, bal_max, pad=1.0)

    origen_opts = sorted(snap["ORIGEN"].dropna().unique().tolist()) if "ORIGEN" in snap.columns else []

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1.2])

    origen_sel = c1.multiselect(
        "Origen (snapshot)",
        options=origen_opts,
        default=origen_opts
    )

    sum_range = c2.slider(
        "Rango Sumergencia (snapshot)",
        min_value=float(s_min), max_value=float(s_max),
        value=(float(s_min), float(s_max))
    )

    est_range = c3.slider(
        "Rango %Estructura (DIN-only)",
        min_value=float(est_min), max_value=float(est_max),
        value=(float(est_min), float(est_max))
    )

    bal_range = c4.slider(
        "Rango %Balance (DIN-only)",
        min_value=float(bal_min), max_value=float(bal_max),
        value=(float(bal_min), float(bal_max))
    )

    # ---------------- Aplicar filtros snapshot ----------------
    snap_f = snap.copy()

    if origen_sel:
        snap_f = snap_f[snap_f["ORIGEN"].isin(origen_sel)]

    snap_f = snap_f[
        (snap_f["Sumergencia"].isna() | snap_f["Sumergencia"].between(sum_range[0], sum_range[1])) &
        (snap_f["%Estructura"].isna() | snap_f["%Estructura"].between(est_range[0], est_range[1])) &
        (snap_f["%Balance"].isna() | snap_f["%Balance"].between(bal_range[0], bal_range[1]))
    ].copy()

    # ---------------- KPIs (incluye NIV) ----------------
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Pozos (snapshot filtrado)", f"{len(snap_f):,}".replace(",", "."))
    k2.metric("Última = DIN", f"{(snap_f['ORIGEN'] == 'DIN').sum():,}".replace(",", ".") if "ORIGEN" in snap_f.columns else "0")
    k3.metric("Última = NIV", f"{(snap_f['ORIGEN'] == 'NIV').sum():,}".replace(",", ".") if "ORIGEN" in snap_f.columns else "0")
    k4.metric("Con Sumergencia", f"{snap_f['Sumergencia'].notna().sum():,}".replace(",", "."))
    k5.metric("Con PB", f"{snap_f['PB'].notna().sum():,}".replace(",", "."))

    # ---------------- Tabla snapshot filtrada ----------------
    st.markdown("### 📋 Pozos (última medición) — filtrados")

    cols_snap = [c for c in [
        "NO_key", "pozo", "ORIGEN", "SE", "DT_plot", "Dias_desde_ultima",

        "PE", "PB", "NM", "NC", "ND",
        "Sumergencia", "Sumergencia_base",

        "AIB Carrera",
        "Sentido giro",
        "Tipo Contrapesos",
        "Distancia contrapesos (cm)",
        "Contrapeso actual",
        "Contrapeso ideal",
        "AIBEB_Torque max contrapeso",
        "Bba Diam Pistón",
        "Bba Llenado",
        "GPM",
        "Caudal bruto efec",
        "Polea Motor",
        "Potencia Motor",
        "RPM Motor",

        "%Estructura", "%Balance",
    ] if c in snap_f.columns]

    df_snap_show = snap_f[cols_snap].copy()
    df_snap_show = df_snap_show.sort_values(["Dias_desde_ultima"], na_position="last")
    st.dataframe(df_snap_show, use_container_width=True, height=360)

    st.divider()

    # ---------------- Gráficos snapshot (DIN y NIV mezclados) ----------------
    st.markdown("### 📈 Gráficos (snapshot, DIN+NIV mezclados)")

    g1, g2 = st.columns(2)

    # 1) Conteo por ORIGEN (snapshot)
    if "ORIGEN" in snap_f.columns and not snap_f.empty:
        c_or = snap_f.groupby("ORIGEN").size().reset_index(name="Pozos")
        fig_or = px.bar(c_or, x="ORIGEN", y="Pozos", title="Pozos por ORIGEN (snapshot)")
        g1.plotly_chart(fig_or, use_container_width=True)
    else:
        g1.info("Sin datos de ORIGEN para graficar.")

    # 2) Histograma de antigüedad
    if "Dias_desde_ultima" in snap_f.columns and snap_f["Dias_desde_ultima"].notna().any():
        fig_age = px.histogram(snap_f, x="Dias_desde_ultima", nbins=30, title="Antigüedad de última medición (días)")
        g2.plotly_chart(fig_age, use_container_width=True)
    else:
        g2.info("No hay antigüedad para graficar.")

    g3, g4 = st.columns(2)

    # 3) Histograma Sumergencia
    sdata = snap_f.dropna(subset=["Sumergencia"]).copy()
    if not sdata.empty:
        fig_s = px.histogram(sdata, x="Sumergencia", nbins=30, title="Distribución de Sumergencia (snapshot)")
        g3.plotly_chart(fig_s, use_container_width=True)
    else:
        g3.info("No hay Sumergencia disponible en el snapshot filtrado.")

    # 4) Histograma PB
    pbdata = snap_f.dropna(subset=["PB"]).copy()
    if not pbdata.empty:
        fig_pb = px.histogram(pbdata, x="PB", nbins=30, title="Distribución de PB (snapshot)")
        g4.plotly_chart(fig_pb, use_container_width=True)
    else:
        g4.info("No hay PB disponible en el snapshot filtrado.")

    st.divider()

    # ---------------- Gráficos DIN-only ----------------
    st.markdown("### 🧰 DIN-only (porque %Estructura/%Balance y Llenado suelen venir de DIN)")

    d1, d2 = st.columns(2)

    eb = snap_f.dropna(subset=["%Estructura", "%Balance"]).copy()
    if not eb.empty:
        fig_eb = px.scatter(
            eb, x="%Estructura", y="%Balance",
            hover_name="NO_key",
            title="%Estructura vs %Balance (snapshot, DIN-only)"
        )
        d1.plotly_chart(fig_eb, use_container_width=True)
        d2.dataframe(
            eb[["NO_key", "ORIGEN", "%Estructura", "%Balance"]].sort_values(["%Estructura"], na_position="last"),
            use_container_width=True,
            height=360
        )
    else:
        d1.info("No hay %Estructura/%Balance suficiente (suelen venir solo de DIN).")
        d2.info("Tabla DIN-only vacía para %Estructura/%Balance.")

    st.divider()

    # ==========================================================
    # Pozos medidos por mes (nunique) -> ETIQUETA + TABLA
    # ==========================================================
    st.markdown("### 🛢️ Pozos medidos por mes (nunique) — tabla")

    df_all_m = df_all.copy()
    df_all_m["DT_plot"] = pd.to_datetime(df_all_m["DT_plot"], errors="coerce")
    df_all_m = df_all_m.dropna(subset=["DT_plot"]).copy()
    df_all_m["Mes"] = df_all_m["DT_plot"].dt.to_period("M").astype(str)

    p_counts = df_all_m.groupby("Mes")["NO_key"].nunique().reset_index(name="Pozos_medidos")
    if not p_counts.empty:
        last_row = p_counts.sort_values("Mes").tail(1)
        last_mes = last_row["Mes"].values[0]
        last_val = int(last_row["Pozos_medidos"].values[0])
        st.write(f"📌 **Último mes ({last_mes})**: **{last_val:,}** pozos medidos".replace(",", "."))
        st.dataframe(p_counts.sort_values("Mes"), use_container_width=True, height=260)
    else:
        st.info("No hay suficientes fechas para armar pozos por mes.")

    st.divider()

    # ==========================================================
    # Cobertura DIN vs NIV (histórico completo) -> con filtro por fecha
    # ==========================================================
    st.markdown("### ✅ Cobertura DIN vs NIV (histórico) — con filtro por fecha")

    if df_all_m.empty:
        st.info("No hay fechas en DT_plot para filtrar cobertura.")
    else:
        dmin = df_all_m["DT_plot"].min()
        dmax = df_all_m["DT_plot"].max()

        cA, cB = st.columns([1.4, 2.6])
        cov_mode = cA.selectbox("Modo de cobertura", ["Última por pozo (snapshot)", "Todas las mediciones (histórico)"], index=1)

        date_range = cB.date_input(
            "Rango de fechas (DT_plot)",
            value=(dmin.date(), dmax.date()),
            min_value=dmin.date(),
            max_value=dmax.date()
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            cov_from = pd.to_datetime(date_range[0])
            cov_to   = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        else:
            cov_from = pd.to_datetime(dmin.date())
            cov_to   = pd.to_datetime(dmax.date()) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        if cov_mode == "Todas las mediciones (histórico)":
            df_cov = df_all.copy()
            df_cov["DT_plot"] = pd.to_datetime(df_cov["DT_plot"], errors="coerce")
            df_cov = df_cov.dropna(subset=["DT_plot"]).copy()
            df_cov = df_cov[(df_cov["DT_plot"] >= cov_from) & (df_cov["DT_plot"] <= cov_to)].copy()

            has_din = set(df_cov[df_cov["ORIGEN"] == "DIN"]["NO_key"].dropna().unique().tolist())
            all_pozos = set(df_cov["NO_key"].dropna().unique().tolist())
            never_din = sorted(list(all_pozos - has_din))

        else:
            df_cov = df_all.copy()
            df_cov["DT_plot"] = pd.to_datetime(df_cov["DT_plot"], errors="coerce")
            df_cov = df_cov.dropna(subset=["DT_plot"]).copy()
            df_cov = df_cov[(df_cov["DT_plot"] >= cov_from) & (df_cov["DT_plot"] <= cov_to)].copy()

            df_cov_sorted = df_cov.sort_values(["NO_key", "DT_plot"], na_position="last")
            snap_cov = df_cov_sorted.groupby("NO_key", as_index=False).tail(1).copy()

            has_din = set(snap_cov[snap_cov["ORIGEN"] == "DIN"]["NO_key"].dropna().unique().tolist())
            all_pozos = set(snap_cov["NO_key"].dropna().unique().tolist())
            never_din = sorted(list(all_pozos - has_din))

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Pozos en ventana", f"{len(all_pozos):,}".replace(",", "."))
        cc2.metric("Pozos con DIN en ventana", f"{len(has_din):,}".replace(",", "."))
        cc3.metric("Pozos sin DIN en ventana", f"{len(never_din):,}".replace(",", "."))

        if never_din:
            with st.expander("Ver lista de pozos sin DIN (en la ventana seleccionada)"):
                st.write(", ".join(never_din))
        else:
            st.caption("No hay pozos 'sin DIN' en la ventana seleccionada.")

    st.divider()

    # ==========================================================
    # Calidad del dato
    # ==========================================================
    st.markdown("### 🧪 Calidad del dato (snapshot filtrado)")

    bad_sum = snap_f[(snap_f["Sumergencia"].notna()) & (snap_f["Sumergencia"] < 0)].copy()

    pb_nonnull = snap_f["PB"].dropna()
    if len(pb_nonnull) >= 10:
        q1v = pb_nonnull.quantile(0.25)
        q3v = pb_nonnull.quantile(0.75)
        iqr = q3v - q1v
        pb_low = q1v - 1.5 * iqr
        pb_high = q3v + 1.5 * iqr
        bad_pb = snap_f[snap_f["PB"].notna() & ((snap_f["PB"] < pb_low) | (snap_f["PB"] > pb_high))].copy()
    else:
        pb_low, pb_high = None, None
        bad_pb = pd.DataFrame()

    q1c, q2c, q3c = st.columns(3)
    q1c.metric("Pozos con Sumergencia < 0", f"{len(bad_sum):,}".replace(",", "."))
    q2c.metric("Pozos con PB anómalo", f"{len(bad_pb):,}".replace(",", "."))
    q3c.metric("Pozos con PB faltante", f"{snap_f['PB'].isna().sum():,}".replace(",", "."))

    if not bad_sum.empty:
        with st.expander("Ver pozos con Sumergencia < 0"):
            cols_bad = [c for c in ["NO_key", "ORIGEN", "DT_plot", "PB", "NM", "NC", "ND", "Sumergencia", "Sumergencia_base"] if c in bad_sum.columns]
            st.dataframe(bad_sum[cols_bad].sort_values(["Sumergencia"], na_position="last"), use_container_width=True, height=320)
    else:
        st.caption("No se detectaron pozos con Sumergencia < 0 en el snapshot filtrado.")

    if not bad_pb.empty:
        with st.expander("Ver pozos con PB anómalo (criterio IQR)"):
            cols_bad2 = [c for c in ["NO_key", "ORIGEN", "DT_plot", "PB", "NM", "NC", "ND", "Sumergencia"] if c in bad_pb.columns]
            st.dataframe(bad_pb[cols_bad2].sort_values(["PB"], na_position="last"), use_container_width=True, height=320)
            if pb_low is not None:
                st.caption(f"Umbrales IQR: PB < {pb_low:.2f} o PB > {pb_high:.2f}")
    else:
        st.caption("No se detectaron pozos con PB anómalo (criterio IQR).")

    st.divider()

    # ==========================================================
    # Pozos con tendencia en aumento (SOLO Estadísticas)
    # ==========================================================
    st.markdown("### 📈 Pozos con tendencia en aumento")

    df_tr = df_all.copy()
    df_tr["DT_plot"] = pd.to_datetime(df_tr["DT_plot"], errors="coerce")
    df_tr = df_tr.dropna(subset=["DT_plot"]).copy()

    trend_var_opts = ["Sumergencia", "PB", "NM", "NC", "ND", "%Estructura", "%Balance", "GPM", "Caudal bruto efec"]
    cT1, cT2, cT3 = st.columns([1.4, 1.0, 1.6])

    trend_var = cT1.selectbox("Variable para tendencia", options=trend_var_opts, index=0)

    min_pts = cT2.slider("Mín. puntos", min_value=2, max_value=20, value=4)
    only_up = cT3.checkbox("Mostrar solo pendiente positiva", value=True)

    if trend_var in ["%Estructura", "%Balance", "GPM", "Caudal bruto efec"] and trend_var not in df_tr.columns:
        if "path" in df_tr.columns:
            df_tr = df_tr.copy()
            df_tr["path_res"] = df_tr["path"].apply(lambda x: resolve_existing_path(x) if pd.notna(x) else None)
            mask_din = (df_tr.get("ORIGEN") == "DIN") & df_tr["path_res"].notna()
            din_paths_all = df_tr.loc[mask_din, "path_res"].astype(str).tolist()

            if din_paths_all:
                df_ex = parse_extras_for_paths(din_paths_all)
                df_ex.index = df_tr.loc[mask_din].index

                for c in EXTRA_FIELDS.keys():
                    if c not in df_tr.columns:
                        df_tr[c] = None
                for c in df_ex.columns:
                    df_tr.loc[mask_din, c] = df_ex[c].values
            else:
                st.info("No hay paths DIN válidos para calcular tendencia de esa variable (DIN-only).")
        else:
            st.info("No existe columna path para poder calcular tendencia DIN-only.")

    if trend_var in df_tr.columns:
        df_tr[trend_var] = pd.to_numeric(df_tr[trend_var], errors="coerce")

    if trend_var not in df_tr.columns:
        st.warning(f"No encuentro la variable '{trend_var}' en el consolidado, y no pude derivarla.")
    else:
        rows = []
        for no, g in df_tr.groupby("NO_key"):
            res = _trend_linear_per_month(g, trend_var)
            if res is None:
                continue
            slope_m, y0, y1, npts = res
            if npts < min_pts:
                continue
            rows.append({
                "NO_key": no,
                "n_puntos": npts,
                "pendiente_por_mes": slope_m,
                "valor_inicial": y0,
                "valor_final": y1,
                "delta_total": (y1 - y0),
                "fecha_inicial": pd.to_datetime(g["DT_plot"].min(), errors="coerce"),
                "fecha_final": pd.to_datetime(g["DT_plot"].max(), errors="coerce"),
            })

        df_trend = pd.DataFrame(rows)

        if df_trend.empty:
            st.info("No hay suficientes pozos que cumplan el mínimo de puntos para calcular tendencia.")
        else:
            if only_up:
                df_trend = df_trend[df_trend["pendiente_por_mes"] > 0].copy()

            df_trend = df_trend.sort_values("pendiente_por_mes", ascending=False)

            t1, t2 = st.columns([1.6, 1.4])

            show_cols = ["NO_key", "n_puntos", "pendiente_por_mes", "delta_total", "valor_inicial", "valor_final", "fecha_inicial", "fecha_final"]
            t1.dataframe(df_trend[show_cols].head(100), use_container_width=True, height=380)

            topn = df_trend.head(30).copy()
            if not topn.empty:
                fig_tr = px.bar(
                    topn.sort_values("pendiente_por_mes", ascending=True),
                    x="pendiente_por_mes",
                    y="NO_key",
                    orientation="h",
                    title=f"Top 30 — Pendiente por mes ({trend_var})"
                )
                t2.plotly_chart(fig_tr, use_container_width=True)
            else:
                t2.info("Sin pozos con pendiente positiva para mostrar.")

    st.divider()

    # ==========================================================
    # (AL FINAL) Semáforo AIB (SE = AIB) — INDEPENDIENTE
    # ==========================================================
    st.divider()
    st.markdown("## 🚦 Semáforo AIB (SE = AIB) — independiente de filtros de Estadísticas")

    aib_base = snap.copy()

    with st.expander("Filtros Semáforo AIB (independientes)", expanded=True):
        aF1, aF2, aF3, aF4 = st.columns([1.2, 1.2, 1.2, 1.2])

        aib_origen_opts = sorted(aib_base["ORIGEN"].dropna().unique().tolist()) if "ORIGEN" in aib_base.columns else []
        aib_origen_sel = aF1.multiselect(
            "Origen (AIB)",
            options=aib_origen_opts,
            default=aib_origen_opts,
            key="aib_origen_sel"
        )

        if aib_base["DT_plot"].notna().any():
            aib_dmin = aib_base["DT_plot"].min()
            aib_dmax = aib_base["DT_plot"].max()
            aib_date_range = aF2.date_input(
                "Rango fechas (DT_plot) — AIB",
                value=(aib_dmin.date(), aib_dmax.date()),
                min_value=aib_dmin.date(),
                max_value=aib_dmax.date(),
                key="aib_date_range"
            )
        else:
            aib_date_range = None
            aF2.info("Sin DT_plot para filtrar por fecha.")

        aib_only_se_aib = aF3.checkbox("Solo SE = AIB", value=True, key="aib_only_se_aib")
        aib_only_with_llen = aF4.checkbox("Solo con Bba Llenado", value=False, key="aib_only_with_llen")

    aib_df = aib_base.copy()

    if aib_origen_sel and "ORIGEN" in aib_df.columns:
        aib_df = aib_df[aib_df["ORIGEN"].isin(aib_origen_sel)].copy()

    if aib_date_range and isinstance(aib_date_range, tuple) and len(aib_date_range) == 2:
        a_from = pd.to_datetime(aib_date_range[0])
        a_to   = pd.to_datetime(aib_date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        aib_df = aib_df[(aib_df["DT_plot"] >= a_from) & (aib_df["DT_plot"] <= a_to)].copy()

    if aib_only_se_aib and "SE" in aib_df.columns:
        aib_df = aib_df[aib_df["SE"].astype(str).str.strip().str.upper().eq("AIB")].copy()

    if aib_only_with_llen and "Bba Llenado" in aib_df.columns:
        aib_df = aib_df[aib_df["Bba Llenado"].notna()].copy()

    st.markdown("### ⚙️ Umbrales Semáforo AIB (independientes)")

    u1, u2, u3, u4 = st.columns([1.2, 1.2, 1.2, 1.2])
    aib_sum_media = u1.number_input("Umbral Sumergencia media (m)", min_value=0.0, max_value=5000.0, value=200.0, step=10.0, key="aib_sum_media")
    aib_sum_alta  = u2.number_input("Umbral Sumergencia alta (m)",  min_value=0.0, max_value=5000.0, value=250.0, step=10.0, key="aib_sum_alta")
    aib_llen_ok   = u3.number_input("Llenado OK (≥ %)",             min_value=0.0, max_value=100.0, value=70.0, step=5.0, key="aib_llen_ok")
    aib_llen_bajo = u4.number_input("Llenado bajo (< %)",           min_value=0.0, max_value=100.0, value=50.0, step=5.0, key="aib_llen_bajo")

    if not aib_df.empty:
        aib_df["Semaforo_AIB"] = aib_df.apply(
            lambda r: compute_semaforo_aib(
                r,
                se_target="AIB",
                sum_media=float(aib_sum_media),
                sum_alta=float(aib_sum_alta),
                llen_ok=float(aib_llen_ok),
                llen_bajo=float(aib_llen_bajo),
            ),
            axis=1
        )
    else:
        aib_df["Semaforo_AIB"] = pd.Series(dtype="string")

    aib_total = (aib_df.get("SE").astype(str).str.upper() == "AIB").sum() if "SE" in aib_df.columns and not aib_df.empty else 0
    aib_ok = (aib_df["Semaforo_AIB"] == "🟢 NORMAL").sum() if "Semaforo_AIB" in aib_df.columns else 0
    aib_alerta = (aib_df["Semaforo_AIB"] == "🟡 ALERTA").sum() if "Semaforo_AIB" in aib_df.columns else 0
    aib_crit = (aib_df["Semaforo_AIB"] == "🔴 CRÍTICO").sum() if "Semaforo_AIB" in aib_df.columns else 0
    aib_sindatos = (aib_df["Semaforo_AIB"] == "SIN DATOS").sum() if "Semaforo_AIB" in aib_df.columns else 0

    kA, kB, kC, kD, kE = st.columns(5)
    kA.metric("Pozos AIB (AIB independiente)", f"{int(aib_total):,}".replace(",", "."))
    kB.metric("🟢 Normal", f"{int(aib_ok):,}".replace(",", "."))
    kC.metric("🟡 Alerta", f"{int(aib_alerta):,}".replace(",", "."))
    kD.metric("🔴 Crítico", f"{int(aib_crit):,}".replace(",", "."))
    kE.metric("Sin datos", f"{int(aib_sindatos):,}".replace(",", "."))

    crit_aib = aib_df[aib_df["Semaforo_AIB"] == "🔴 CRÍTICO"].copy() if "Semaforo_AIB" in aib_df.columns else pd.DataFrame()

    if not crit_aib.empty:
        st.markdown("#### 🔴 AIB Crítico — prioridad (independiente)")
        cols_crit = [c for c in [
            "NO_key","pozo","ORIGEN","DT_plot","Dias_desde_ultima","SE",
            "PB","Sumergencia","Bba Llenado","Sumergencia_base",
            "%Estructura","%Balance","GPM","Caudal bruto efec",
            "Semaforo_AIB"
        ] if c in crit_aib.columns]
        crit_aib = crit_aib.sort_values(["Sumergencia","Bba Llenado"], ascending=[False, True], na_position="last")
        st.dataframe(crit_aib[cols_crit], use_container_width=True, height=320)
    else:
        st.caption("No hay pozos en 🔴 CRÍTICO con los umbrales actuales (en el AIB independiente).")

    st.markdown("### 📋 Semáforo AIB — tabla (independiente)")
    cols_aib = [c for c in [
        "NO_key","pozo","ORIGEN","DT_plot","Dias_desde_ultima","SE",
        "PB","NM","NC","ND","Sumergencia","Sumergencia_base","Bba Llenado",
        "%Estructura","%Balance","GPM","Caudal bruto efec",
        "Semaforo_AIB"
    ] if c in aib_df.columns]
    if not aib_df.empty and cols_aib:
        st.dataframe(
            aib_df[cols_aib].sort_values(["Semaforo_AIB","Dias_desde_ultima"], na_position="last"),
            use_container_width=True,
            height=420
        )
    else:
        st.info("No hay datos para mostrar en Semáforo AIB (independiente) con los filtros actuales.")


# ==========================================================
# TAB 3: MAPA DE SUMERGENCIA (HEATMAP DENSIDAD)
# ==========================================================
with tab_map:
    st.subheader("🗺️ Mapa de sumergencia (heatmap densidad — última medición por pozo)")

    snap_map = build_last_snapshot_for_map(din_ok, niv_ok)

    if snap_map.empty:
        st.info("No hay datos suficientes para armar el mapa.")
        st.stop()
    
    snap_map = snap_map.copy()
    snap_map["DT_plot"] = pd.to_datetime(snap_map["DT_plot"], errors="coerce")
    snap_map = snap_map.dropna(subset=["DT_plot"]).copy()


    now = pd.Timestamp.now()
    snap_map["Dias_desde_ultima"] = (now - snap_map["DT_plot"]).dt.total_seconds() / 86400.0
    snap_map["Sumergencia"] = pd.to_numeric(snap_map.get("Sumergencia"), errors="coerce")

    
    
    # Cargar coordenadas del repo
    coords = load_coords_repo()
    
    
    if coords.empty:
        st.error(
            "No encontré el Excel de coordenadas en el repo.\n\n"
            "Debe existir en: assets/Nombres-Pozo_con_coordenadas.xlsx\n"
            "y tener columnas: nombre_corto, GEO_LATITUDE, GEO_LONGITUDE"
        )
        st.stop()

    coords = coords.copy()
    coords["NO_key"] = coords["nombre_corto"].apply(normalize_no_exact)
    snap_map["NO_key"] = snap_map["NO_key"].apply(normalize_no_exact)

    m = snap_map.merge(
        coords[["NO_key", "nombre_corto", "GEO_LATITUDE", "GEO_LONGITUDE"]],
        on="NO_key",
        how="left"
    ).rename(columns={"GEO_LATITUDE": "lat", "GEO_LONGITUDE": "lon"})

    m["lat"] = pd.to_numeric(m["lat"], errors="coerce")
    m["lon"] = pd.to_numeric(m["lon"], errors="coerce")

    st.markdown("### Filtros")
    f2, f4 = st.columns([2.0, 1.4])
    
    m_f = m.copy()
    
    # SIEMPRE solo con coordenadas (sin checkbox)
    m_f = m_f[m_f["lat"].notna() & m_f["lon"].notna()].copy()


    s_ok = m_f["Sumergencia"].dropna()
    if s_ok.empty:
        st.info("No hay Sumergencia numérica en la última medición (snapshot) para mapear con estos filtros.")
        st.stop()

    smin = float(s_ok.min())
    smax = float(s_ok.max())
    if smin == smax:
        smin, smax = smin - 1.0, smax + 1.0

    sum_range_map = f2.slider(
        "Rango de Sumergencia (última)",
        min_value=float(smin),
        max_value=float(smax),
        value=(float(smin), float(smax)),
        key="map_sum_range"
    )

    # filtro por dias
    # filtro por dias (forzar numérico SIEMPRE antes)
    m_f["Dias_desde_ultima"] = pd.to_numeric(m_f.get("Dias_desde_ultima"), errors="coerce")
    
    d_ok = m_f["Dias_desde_ultima"].dropna()
    if not d_ok.empty:
        dmin = float(d_ok.min())
        dmax = float(d_ok.max())
        if dmin == dmax:
            dmin, dmax = dmin - 1.0, dmax + 1.0
    
        dias_range = f4.slider(
            "Rango días desde última",
            min_value=float(dmin),
            max_value=float(dmax),
            value=(float(dmin), float(dmax)),
            key="map_dias_range"
        )
    
        # aplicar filtro
        m_f = m_f[m_f["Dias_desde_ultima"].between(dias_range[0], dias_range[1], inclusive="both")].copy()

    m_f = m_f[m_f["Sumergencia"].between(sum_range_map[0], sum_range_map[1], inclusive="both")].copy()

    if m_f.empty:
        st.warning("No quedaron pozos con los filtros seleccionados.")
        st.stop()


    # ---------- (FIX) DF "delgado" y JSON-safe para pydeck ----------
    m_map = m_f.copy()
    
    # Tipos numéricos limpios
    m_map["lat"] = pd.to_numeric(m_map["lat"], errors="coerce")
    m_map["lon"] = pd.to_numeric(m_map["lon"], errors="coerce")
    m_map["Sumergencia"] = pd.to_numeric(m_map.get("Sumergencia"), errors="coerce")
    m_map["Dias_desde_ultima"] = pd.to_numeric(m_map.get("Dias_desde_ultima"), errors="coerce")
    
    # Convertir datetime a string para tooltip (pydeck no banca Timestamp/NaT en JSON)
    if "DT_plot" in m_map.columns:
        m_map["DT_plot_str"] = pd.to_datetime(m_map["DT_plot"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    else:
        m_map["DT_plot_str"] = None
    
    # Reemplazar NaN/NaT por None (JSON-safe)
    m_map = m_map.replace({pd.NA: None})
    m_map = m_map.where(pd.notnull(m_map), None)
    
    # Quedarse SOLO con columnas necesarias para mapa+tooltip
    keep_cols = [c for c in ["NO_key", "ORIGEN", "DT_plot_str", "Sumergencia", "Dias_desde_ultima", "lat", "lon"] if c in m_map.columns]
    m_map = m_map[keep_cols].copy()

    
    # Heatmap densidad
    # Heatmap densidad
    import pydeck as pdk

    center_lat = float(m_map["lat"].mean()) if m_map["lat"].notna().any() else -45.0
    center_lon = float(m_map["lon"].mean()) if m_map["lon"].notna().any() else -68.0

    heat = pdk.Layer(
        "HeatmapLayer",
        data=m_map,
        get_position='[lon, lat]',
        get_weight="Sumergencia",
        radiusPixels=45,
        intensity=1.0,
        threshold=0.05,
    )

    pts = pdk.Layer(
        "ScatterplotLayer",
        data=m_map,
        get_position='[lon, lat]',
        get_radius=120,
        pickable=True,
    )

    tooltip = {
        "html": (
            "<b>Pozo:</b> {NO_key}<br/>"
            "<b>Origen:</b> {ORIGEN}<br/>"
            "<b>DT:</b> {DT_plot_str}<br/>"
            "<b>Sumergencia:</b> {Sumergencia}<br/>"
            "<b>Días desde última:</b> {Dias_desde_ultima}"
        )
    }

    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=9)

    st.pydeck_chart(
        pdk.Deck(
            layers=[heat, pts],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style=None,
        ),
        use_container_width=True
    )

    st.divider()


    st.markdown("### 📋 Pozos filtrados (selección y exportación)")

    show_cols = [c for c in [
        "NO_key", "ORIGEN", "DT_plot", "Dias_desde_ultima", "Sumergencia",
        "PE", "PB", "NM", "NC", "ND", "Sumergencia_base",
        "lat", "lon",
    ] if c in m_f.columns]

    t = m_f[show_cols].copy()
    t = t.sort_values(["Sumergencia"], ascending=False, na_position="last").reset_index(drop=True)
    t.insert(0, "Seleccionar", False)

    edited = st.data_editor(
        t,
        use_container_width=True,
        height=380,
        hide_index=True
    )

    picked = edited[edited["Seleccionar"] == True].drop(columns=["Seleccionar"], errors="ignore").copy()
    st.caption(f"Seleccionados: {len(picked)}")

    csv_bytes = picked.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar seleccionados (CSV)",
        data=csv_bytes,
        file_name="pozos_sumergencia_seleccionados.csv",
        mime="text/csv"
    )

    # Excel (si tenés openpyxl instalado)
    try:
        import io
        buf = io.BytesIO()
        picked.to_excel(buf, index=False, sheet_name="seleccionados")
        st.download_button(
            "⬇️ Descargar seleccionados (Excel)",
            data=buf.getvalue(),
            file_name="pozos_sumergencia_seleccionados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception:
        st.info("Para exportar a Excel, agregá `openpyxl` a requirements.txt (CSV ya funciona).")

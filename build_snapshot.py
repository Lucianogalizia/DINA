# ==========================================================
# build_snapshot.py
# Job nocturno — corre una vez por día (~12:30am)
# ==========================================================

import os
import re
import tempfile
import time
from pathlib import Path

import pandas as pd

# ── Config ──────────────────────────────────────────────────────────
GCS_BUCKET = os.environ.get("DINAS_BUCKET", "").strip()
GCS_PREFIX = os.environ.get("DINAS_GCS_PREFIX", "").strip().strip("/")

SNAPSHOT_BLOB = "snapshot.parquet"
if GCS_PREFIX:
    SNAPSHOT_BLOB = f"{GCS_PREFIX}/{SNAPSHOT_BLOB}"

EXTRA_FIELDS = {
    "Tipo AIB":                    ("AIB",        "MA"),
    "AIB Carrera":                 ("AIB",        "CS"),
    "Sentido giro":                ("AIB",        "SG"),
    "Tipo Contrapesos":            ("CONTRAPESO", "TP"),
    "Distancia contrapesos (cm)":  ("CONTRAPESO", "DE"),
    "Contrapeso actual":           ("RARE",       "CA"),
    "Contrapeso ideal":            ("RARE",       "CM"),
    "AIBEB_Torque max contrapeso": ("RAEB",       "TM"),
    "%Estructura":                 ("RARE",       "SE"),
    "%Balance":                    ("RARR",       "PC"),
    "Bba Diam Pistón":             ("BOMBA",      "DP"),
    "Bba Prof":                    ("BOMBA",      "PB"),
    "Bba Llenado":                 ("BOMBA",      "CA"),
    "GPM":                         ("AIB",        "GM"),
    "Caudal bruto efec":           ("RBO",        "CF"),
    "Polea Motor":                 ("MOTOR",      "DP"),
    "Potencia Motor":              ("MOTOR",      "PN"),
    "RPM Motor":                   ("MOTOR",      "RM"),
}

SECTION_RE = re.compile(r"^\s*\[(.+?)\]\s*$")
KV_RE      = re.compile(r"^\s*([^=]+?)\s*=\s*(.*?)\s*$")


# ── GCS helpers ─────────────────────────────────────────────────────
def _gcs_client():
    from google.cloud import storage
    return storage.Client()


def _parse_gs_url(gs_url: str):
    u = gs_url.strip()[5:]
    bucket, _, blob = u.partition("/")
    return bucket, blob


def _download_to_temp(gs_url: str) -> str:
    client = _gcs_client()
    bucket_name, blob_name = _parse_gs_url(gs_url)
    safe_name = blob_name.replace("/", "__")
    local_path = os.path.join(tempfile.gettempdir(), safe_name)

    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    client.bucket(bucket_name).blob(blob_name).download_to_filename(local_path)
    return local_path


def _upload_parquet(df: pd.DataFrame, blob_name: str):
    """Sube un DataFrame como parquet a GCS (blindado contra errores de tipos)."""

    client = _gcs_client()
    local  = os.path.join(tempfile.gettempdir(), "snapshot_tmp.parquet")

    # 🔧 FIX GLOBAL:
    # Convertir columnas object mezcladas a string para evitar errores ArrowTypeError
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("string").fillna("")

    df.to_parquet(local, index=False)

    client.bucket(GCS_BUCKET).blob(blob_name).upload_from_filename(local)
    print(f"  ✅ Subido: gs://{GCS_BUCKET}/{blob_name}  ({len(df)} filas)")


# ── Parse helpers ───────────────────────────────────────────────────
def _read_text(path: str) -> str:
    p = Path(path)
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return p.read_text(encoding=enc, errors="strict")
        except Exception:
            pass
    return p.read_text(encoding="latin-1", errors="ignore")


def _safe_float(v):
    if v is None:
        return None
    s = str(v).strip().replace(",", ".")
    if "=" in s:
        s = s.split("=")[-1].strip()
    try:
        return float(s)
    except Exception:
        return None


def _parse_extras(path_str: str) -> dict:
    try:
        txt = _read_text(path_str)
    except Exception:
        return {k: None for k in EXTRA_FIELDS}

    wanted = {(sec.upper(), key.upper()): col for col, (sec, key) in EXTRA_FIELDS.items()}
    out = {col: None for col in EXTRA_FIELDS}
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
            fv = _safe_float(v)
            out[col] = fv if fv is not None else (v if v != "" else None)

    return out


# ── Main ─────────────────────────────────────────────────────────────
def build_snapshot():

    if not GCS_BUCKET:
        print("❌ DINAS_BUCKET no está configurado.")
        return

    print("=" * 55)
    print("DINA — build_snapshot.py")
    print("=" * 55)

    blob_index = f"{GCS_PREFIX}/din_index.parquet" if GCS_PREFIX else "din_index.parquet"
    local_idx  = _download_to_temp(f"gs://{GCS_BUCKET}/{blob_index}")
    din_index  = pd.read_parquet(local_idx)

    if "NO_key" not in din_index.columns:
        print("❌ No se encontró NO_key")
        return

    snap = din_index.groupby("NO_key", as_index=False).tail(1).copy()

    extras_rows = []
    for _, row in snap.iterrows():
        try:
            local = _download_to_temp(row["path"])
            extras = _parse_extras(local)
        except Exception:
            extras = {k: None for k in EXTRA_FIELDS}
        extras_rows.append(extras)

    df_extras = pd.DataFrame(extras_rows, index=snap.index)
    snapshot  = pd.concat([snap.reset_index(drop=True),
                           df_extras.reset_index(drop=True)], axis=1)

    print("\n📤 Subiendo snapshot.parquet...")
    _upload_parquet(snapshot, SNAPSHOT_BLOB)

    print("✅ Terminado")


if __name__ == "__main__":
    build_snapshot()

# ==========================================================
# build_snapshot.py
# Job nocturno — corre una vez por día (~12:30am)
#
# Qué hace:
#   1. Lee din_index.parquet desde GCS
#   2. Para cada pozo, baja el .din más reciente y extrae los extras
#   3. Guarda snapshot.parquet en GCS
#
# La app.py solo lee ese parquet → carga en 3 segundos.
#
# Cómo correr manualmente:
#   python build_snapshot.py
#
# En Cloud Scheduler / Cloud Run Job:
#   CMD ["python", "build_snapshot.py"]
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

SNAPSHOT_BLOB = "snapshot.parquet"  # destino en GCS
if GCS_PREFIX:
    SNAPSHOT_BLOB = f"{GCS_PREFIX}/{SNAPSHOT_BLOB}"

# Campos a extraer de cada .din
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
    u = gs_url.strip()[5:]          # quitar "gs://"
    bucket, _, blob = u.partition("/")
    return bucket, blob


def _download_to_temp(gs_url: str) -> str:
    """Baja un archivo de GCS a /tmp y devuelve el path local."""
    client = _gcs_client()
    bucket_name, blob_name = _parse_gs_url(gs_url)
    safe_name  = blob_name.replace("/", "__")
    local_path = os.path.join(tempfile.gettempdir(), safe_name)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    client.bucket(bucket_name).blob(blob_name).download_to_filename(local_path)
    return local_path


def _upload_parquet(df: pd.DataFrame, blob_name: str):
    """Sube un DataFrame como parquet a GCS."""
    client = _gcs_client()
    local  = os.path.join(tempfile.gettempdir(), "snapshot_tmp.parquet")

    # 🔧 FIX: esta columna puede venir mezclada (texto + NaN/float) y pyarrow rompe
    if "Tipo Contrapesos" in df.columns:
        df["Tipo Contrapesos"] = df["Tipo Contrapesos"].astype("string").fillna("")

    df.to_parquet(local, index=False)
    client.bucket(GCS_BUCKET).blob(blob_name).upload_from_filename(local)
    print(f"  ✅ Subido: gs://{GCS_BUCKET}/{blob_name}  ({len(df)} filas)")


# ── Parseo .din ─────────────────────────────────────────────────────
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
    """Extrae los campos EXTRA_FIELDS de un archivo .din."""
    try:
        txt = _read_text(path_str)
    except Exception:
        return {k: None for k in EXTRA_FIELDS}

    wanted  = {(sec.upper(), key.upper()): col for col, (sec, key) in EXTRA_FIELDS.items()}
    out     = {col: None for col in EXTRA_FIELDS}
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
            fv  = _safe_float(v)
            out[col] = fv if fv is not None else (v if v != "" else None)

    # Fallback %Balance
    if out.get("%Balance") is None:
        section = None
        for line in txt.splitlines():
            m = SECTION_RE.match(line)
            if m:
                section = m.group(1).strip().upper()
                continue
            m = KV_RE.match(line)
            if not m or not section:
                continue
            if section == "RARR" and m.group(1).strip().upper() == "PC":
                fv = _safe_float(m.group(2).strip())
                out["%Balance"] = fv
                break

    return out


def _map_to_gcs(path_str: str) -> str | None:
    """Convierte path local data_store/... a gs://bucket/..."""
    if not path_str or not GCS_BUCKET:
        return None
    p   = str(path_str).replace("\\", "/")
    idx = p.lower().find("/data_store/")
    if idx == -1:
        return None
    rel = p[idx + 1:]
    if GCS_PREFIX:
        return f"gs://{GCS_BUCKET}/{GCS_PREFIX}/{rel}"
    return f"gs://{GCS_BUCKET}/{rel}"


# ── Main ─────────────────────────────────────────────────────────────
def build_snapshot():
    if not GCS_BUCKET:
        print("❌ DINAS_BUCKET no está configurado.")
        return

    print("=" * 55)
    print("  DINA — build_snapshot.py")
    print("=" * 55)
    t0 = time.time()

    # 1. Leer din_index.parquet
    print("\n📥 Leyendo din_index.parquet...")
    blob_index = f"{GCS_PREFIX}/din_index.parquet" if GCS_PREFIX else "din_index.parquet"
    local_idx  = _download_to_temp(f"gs://{GCS_BUCKET}/{blob_index}")
    din_index  = pd.read_parquet(local_idx)
    print(f"   {len(din_index)} filas en el índice")

    if "NO_key" not in din_index.columns:
        # Intentar derivar NO_key desde columna pozo/NO
        for cand in ["pozo", "NO"]:
            if cand in din_index.columns:
                din_index["NO_key"] = din_index[cand].astype(str).str.strip()
                break

    if "NO_key" not in din_index.columns:
        print("❌ No se encontró columna NO_key en din_index.parquet")
        return

    # 2. Para cada pozo, quedarnos con la última medición
    sort_cols = [c for c in ["din_datetime", "mtime"] if c in din_index.columns]
    if sort_cols:
        din_index = din_index.sort_values(sort_cols, na_position="last")

    snap = din_index.groupby("NO_key", as_index=False).tail(1).copy()
    print(f"   {len(snap)} pozos únicos (última medición por pozo)")

    # 3. Resolver paths a GCS
    if "path" in snap.columns:
        def _resolve(p):
            if not p or pd.isna(p):
                return None
            s = str(p).strip()
            if s.lower().startswith("gs://"):
                return s
            return _map_to_gcs(s)
        snap["path_gs"] = snap["path"].apply(_resolve)
    else:
        snap["path_gs"] = None

    # 4. Descargar y parsear cada .din
    print(f"\n⚙️  Procesando {len(snap)} archivos .din...")
    extras_rows = []
    ok = 0
    err = 0

    for i, (_, row) in enumerate(snap.iterrows()):
        path_gs = row.get("path_gs")
        no_key  = row.get("NO_key", "?")

        if not path_gs:
            extras_rows.append({k: None for k in EXTRA_FIELDS})
            err += 1
            continue

        try:
            local = _download_to_temp(path_gs)
            extras = _parse_extras(local)
            extras_rows.append(extras)
            ok += 1
        except Exception:
            extras_rows.append({k: None for k in EXTRA_FIELDS})
            err += 1

        # Progreso cada 50
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (len(snap) - i - 1)
            print(f"   {i+1}/{len(snap)}  ✅{ok}  ❌{err}  ETA: {int(eta)}s")

    print(f"\n   Listo: ✅ {ok} ok  ❌ {err} sin path/error")

    # 5. Combinar snap + extras
    df_extras = pd.DataFrame(extras_rows, index=snap.index)
    snapshot  = pd.concat([snap.reset_index(drop=True), df_extras.reset_index(drop=True)], axis=1)

    # 6. Calcular Sumergencia
    def _sumergencia(row):
        pb = _safe_float(row.get("PB"))
        if pb is None:
            return None, None
        for col in ["NC", "NM", "ND"]:
            v = _safe_float(row.get(col))
            if v is not None:
                return pb - v, col
        return None, None

    tmp = snapshot.apply(_sumergencia, axis=1, result_type="expand")
    snapshot["Sumergencia"]      = tmp[0]
    snapshot["Sumergencia_base"] = tmp[1]

    # Antigüedad
    now = pd.Timestamp.now()
    if sort_cols:
        snapshot["DT_plot"] = pd.to_datetime(snapshot[sort_cols[0]], errors="coerce")
    else:
        snapshot["DT_plot"] = pd.NaT

    snapshot["Dias_desde_ultima"] = (now - snapshot["DT_plot"]).dt.total_seconds() / 86400.0

    # 7. Subir a GCS
    print(f"\n📤 Subiendo snapshot.parquet...")
    _upload_parquet(snapshot, SNAPSHOT_BLOB)

    total = int(time.time() - t0)
    print(f"\n✅ Terminado en {total // 60}m {total % 60}s")
    print("=" * 55)


if __name__ == "__main__":
    build_snapshot()

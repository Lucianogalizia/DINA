# ==========================================================
# diagnostico_tab.py
# Pestaña "Diagnósticos" — Análisis IA de cartas dinamométricas
#
# Modelo: gpt-5.2-chat-latest (OpenAI)
# Caché:  GCS → diagnosticos/{NO_key}/diagnostico.json
# Lógica: analiza hasta 3 DINs más recientes por pozo.
#         Regenera automáticamente si hay un DIN más nuevo
#         que la fecha del último diagnóstico guardado.
#
# API Key: GCP Secret Manager → secret "OPENAI_API_KEY"
#          (fallback: variable de entorno OPENAI_API_KEY)
#
# v4: Botón "Generar todos" con barra de progreso
#     Estados simplificados → solo ACTIVA / RESUELTA
# ==========================================================

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------ #
#  Catálogo base de problemáticas
# ------------------------------------------------------------------ #
CATALOGO_PROBLEMATICAS = [
    "Llenado bajo de bomba",
    "Golpeo de fondo",
    "Fuga en válvula viajera",
    "Fuga en válvula fija",
    "Interferencia de fluido",
    "Bomba asentada parcialmente",
    "Gas en bomba",
    "Desbalance de contrapesos",
    "Sobrecarga estructural",
    "Subcarrera / carrera insuficiente",
    "Desgaste de bomba",
    "Sumergencia crítica",
    "Tendencia de declinación de caudal",
    "Rotura / desgaste de varillas",
    "Exceso de fricción en varillas",
]

SEVERIDAD_ORDEN = {"CRÍTICA": 0, "ALTA": 1, "MEDIA": 2, "BAJA": 3}

SEVERIDAD_COLOR = {
    "BAJA":    "#28a745",
    "MEDIA":   "#ffc107",
    "ALTA":    "#fd7e14",
    "CRÍTICA": "#dc3545",
}
SEVERIDAD_EMOJI = {
    "BAJA":    "🟢",
    "MEDIA":   "🟡",
    "ALTA":    "🟠",
    "CRÍTICA": "🔴",
}
ESTADO_EMOJI = {
    "ACTIVA":   "⚠️",
    "RESUELTA": "✅",
}
ESTADO_COLOR = {
    "ACTIVA":   "#dc3545",
    "RESUELTA": "#28a745",
}


# ------------------------------------------------------------------ #
#  Helpers GCS
# ------------------------------------------------------------------ #

def _get_gcs_client():
    try:
        from google.cloud import storage
        return storage.Client()
    except Exception:
        return None


def _get_openai_key() -> str | None:
    try:
        from google.cloud import secretmanager
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
        secret_name = os.environ.get("OPENAI_SECRET_NAME", "OPENAI_API_KEY")
        if project_id:
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            key = response.payload.data.decode("UTF-8").strip()
            if key:
                return key
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY", "").strip() or None


# ------------------------------------------------------------------ #
#  Caché GCS
# ------------------------------------------------------------------ #

def _load_diag_from_gcs(bucket_name: str, no_key: str, prefix: str = "") -> dict | None:
    client = _get_gcs_client()
    if not client:
        return None
    blob_name = f"diagnosticos/{no_key}/diagnostico.json"
    if prefix:
        blob_name = f"{prefix}/{blob_name}"
    try:
        bucket = client.bucket(bucket_name)
        blob   = bucket.blob(blob_name)
        if not blob.exists():
            return None
        return json.loads(blob.download_as_text(encoding="utf-8"))
    except Exception:
        return None


def _save_diag_to_gcs(bucket_name: str, no_key: str, diag: dict, prefix: str = "") -> bool:
    client = _get_gcs_client()
    if not client:
        return False
    blob_name = f"diagnosticos/{no_key}/diagnostico.json"
    if prefix:
        blob_name = f"{prefix}/{blob_name}"
    try:
        bucket = client.bucket(bucket_name)
        blob   = bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(diag, ensure_ascii=False, indent=2, default=str),
            content_type="application/json"
        )
        return True
    except Exception:
        return False


def _load_all_diags_from_gcs(bucket_name: str, pozos: list[str], prefix: str) -> dict[str, dict]:
    client = _get_gcs_client()
    if not client:
        return {}
    results = {}
    bucket  = client.bucket(bucket_name)
    for no_key in pozos:
        blob_name = f"diagnosticos/{no_key}/diagnostico.json"
        if prefix:
            blob_name = f"{prefix}/{blob_name}"
        try:
            blob = bucket.blob(blob_name)
            if blob.exists():
                data = json.loads(blob.download_as_text(encoding="utf-8"))
                if "error" not in data:
                    results[no_key] = data
        except Exception:
            pass
    return results


# ------------------------------------------------------------------ #
#  Parseo de .din
# ------------------------------------------------------------------ #

def _read_text(path: str) -> str:
    p = Path(path)
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return p.read_text(encoding=enc, errors="strict")
        except Exception:
            pass
    return p.read_text(encoding="latin-1", errors="ignore")


def _parse_din_full(path_str: str) -> dict:
    import re
    SECTION_RE = re.compile(r"^\s*\[(.+?)\]\s*$")
    KV_RE      = re.compile(r"^\s*([^=]+?)\s*=\s*(.*?)\s*$")
    POINT_RE   = re.compile(r"^(X|Y)\s*(\d+)$", re.IGNORECASE)

    txt      = _read_text(path_str)
    sections: dict[str, dict] = {}
    section  = None
    xs: dict[int, float] = {}
    ys: dict[int, float] = {}
    in_cs    = False

    for line in txt.splitlines():
        m = SECTION_RE.match(line)
        if m:
            section = m.group(1).strip().upper()
            in_cs   = (section == "CS")
            sections.setdefault(section, {})
            continue

        m = KV_RE.match(line)
        if not m or not section:
            continue

        k = m.group(1).strip()
        v = m.group(2).strip()

        if in_cs:
            mp = POINT_RE.match(k)
            if mp:
                xy  = mp.group(1).upper()
                idx = int(mp.group(2))
                try:
                    val = float(v.replace(",", "."))
                except Exception:
                    continue
                (xs if xy == "X" else ys)[idx] = val
                continue

        sections[section][k] = v

    idxs      = sorted(set(xs) & set(ys))
    cs_points = [{"X": xs[i], "Y": ys[i]} for i in idxs]
    return {"sections": sections, "cs_points": cs_points}


def _safe_float(v) -> float | None:
    if v is None:
        return None
    s = str(v).strip().replace(",", ".")
    if "=" in s:
        s = s.split("=")[-1].strip()
    try:
        return float(s)
    except Exception:
        return None


def _extract_variables(parsed: dict) -> dict:
    secs = parsed.get("sections", {})

    def g(sec: str, key: str):
        return secs.get(sec.upper(), {}).get(key)

    v = {
        "NO":                g("GEN", "NO"),
        "FE":                g("GEN", "FE"),
        "HO":                g("GEN", "HO"),
        "Tipo_AIB":          g("AIB", "MA"),
        "Carrera_pulg":      _safe_float(g("AIB", "CS")),
        "Golpes_min":        _safe_float(g("AIB", "GM")),
        "Sentido_giro":      g("AIB", "SG"),
        "Tipo_contrapeso":   g("CONTRAPESO", "TP"),
        "Dist_contrapeso":   _safe_float(g("CONTRAPESO", "DE")),
        "Polea_motor":       _safe_float(g("MOTOR", "DP")),
        "Potencia_motor":    _safe_float(g("MOTOR", "PN")),
        "RPM_motor":         _safe_float(g("MOTOR", "RM")),
        "Diam_piston_pulg":  _safe_float(g("BOMBA", "DP")),
        "Prof_bomba_m":      _safe_float(g("BOMBA", "PB")),
        "Llenado_pct":       _safe_float(g("BOMBA", "CA")),
        "PE_m":              _safe_float(g("NIV", "PE")),
        "PB_m":              _safe_float(g("NIV", "PB")),
        "NM_m":              _safe_float(g("NIV", "NM")),
        "NC_m":              _safe_float(g("NIV", "NC")),
        "ND_m":              _safe_float(g("NIV", "ND")),
        "Contrapeso_actual": _safe_float(g("RARE", "CA")),
        "Contrapeso_ideal":  _safe_float(g("RARE", "CM")),
        "Pct_estructura":    _safe_float(g("RARE", "SE")),
        "Pct_balance":       _safe_float(g("RARR", "PC")),
        "Caudal_bruto":      _safe_float(g("RBO", "CF")),
        "Torque_max":        _safe_float(g("RAEB", "TM")),
    }

    pb = v.get("Prof_bomba_m")
    for nk in ["NC_m", "NM_m", "ND_m"]:
        nv = v.get(nk)
        if pb is not None and nv is not None:
            v["Sumergencia_m"]    = round(pb - nv, 1)
            v["Base_sumergencia"] = nk.replace("_m", "")
            break
    else:
        v["Sumergencia_m"]    = None
        v["Base_sumergencia"] = None

    return v


def _describe_cs_shape(cs_points: list[dict]) -> str:
    if not cs_points:
        return "Sin datos de carta de superficie [CS]."

    xs = [p["X"] for p in cs_points]
    ys = [p["Y"] for p in cs_points]
    n  = len(cs_points)

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    carrera      = round(x_max - x_min, 1)
    rango_carga  = round(y_max - y_min, 1)

    area = 0.0
    for i in range(n):
        j     = (i + 1) % n
        area += xs[i] * ys[j]
        area -= xs[j] * ys[i]
    area = round(abs(area) / 2.0, 1)

    rect_area  = carrera * rango_carga
    fill_ratio = round(area / rect_area, 2) if rect_area > 0 else 0
    forma      = "compacta/llena" if fill_ratio > 0.55 else "delgada/estrecha"

    idx_max     = ys.index(max(ys))
    idx_min     = ys.index(min(ys))
    pos_max_pct = round((xs[idx_max] - x_min) / (carrera or 1) * 100, 1)
    pos_min_pct = round((xs[idx_min] - x_min) / (carrera or 1) * 100, 1)

    pts_sorted   = sorted(cs_points, key=lambda p: p["X"])
    n_cuart      = max(2, int(n * 0.25))
    sub_pts      = pts_sorted[:n_cuart]
    baj_pts      = pts_sorted[-n_cuart:]
    delta_subida = round(sub_pts[-1]["Y"] - sub_pts[0]["Y"], 1)
    delta_bajada = round(baj_pts[-1]["Y"] - baj_pts[0]["Y"], 1)

    return (
        f"n_puntos={n} | carrera_efectiva={carrera} | "
        f"carga_max={round(y_max,1)} | carga_min={round(y_min,1)} | rango_carga={rango_carga} | "
        f"area={area} | fill_ratio={fill_ratio} | forma={forma} | "
        f"pos_carga_max={pos_max_pct}%_carrera | pos_carga_min={pos_min_pct}%_carrera | "
        f"delta_y_subida_inicial={delta_subida} | delta_y_bajada_final={delta_bajada}"
    )


# ------------------------------------------------------------------ #
#  Construcción del prompt
# ------------------------------------------------------------------ #

def _build_prompt(no_key: str, mediciones: list[dict]) -> str:
    catalogo_str = "\n".join(f"  - {p}" for p in CATALOGO_PROBLEMATICAS)
    lineas_med   = []
    vars_primera = None

    for i, m in enumerate(mediciones):
        label = "Única medición" if len(mediciones) == 1 else ["Más antigua", "Intermedia", "Más reciente"][min(i, 2)]
        lineas_med.append(f"\n### [{label}] Fecha: {m['fecha']}")
        v = m["vars"]
        if vars_primera is None:
            vars_primera = v

        lineas_med.append(
            f"  Tipo AIB: {v.get('Tipo_AIB') or 'N/D'} | "
            f"Carrera: {v.get('Carrera_pulg') or 'N/D'} pulg | "
            f"Golpes/min: {v.get('Golpes_min') or 'N/D'} | "
            f"Sentido giro: {v.get('Sentido_giro') or 'N/D'}"
        )
        lineas_med.append(
            f"  Motor: {v.get('Potencia_motor') or 'N/D'} HP | "
            f"RPM: {v.get('RPM_motor') or 'N/D'} | "
            f"Polea: {v.get('Polea_motor') or 'N/D'}"
        )
        lineas_med.append(
            f"  Bomba: Ø pistón {v.get('Diam_piston_pulg') or 'N/D'} pulg | "
            f"Prof: {v.get('Prof_bomba_m') or 'N/D'} m | "
            f"Llenado: {v.get('Llenado_pct') or 'N/D'}%"
        )
        lineas_med.append(
            f"  Niveles → PE: {v.get('PE_m') or 'N/D'} m | "
            f"PB: {v.get('PB_m') or 'N/D'} m | "
            f"NM: {v.get('NM_m') or 'N/D'} m | "
            f"NC: {v.get('NC_m') or 'N/D'} m | "
            f"ND: {v.get('ND_m') or 'N/D'} m"
        )
        lineas_med.append(
            f"  Sumergencia: {v.get('Sumergencia_m') or 'N/D'} m "
            f"(base: {v.get('Base_sumergencia') or 'N/D'})"
        )
        lineas_med.append(
            f"  Contrapeso actual: {v.get('Contrapeso_actual') or 'N/D'} | "
            f"ideal: {v.get('Contrapeso_ideal') or 'N/D'} | "
            f"%Balance: {v.get('Pct_balance') or 'N/D'} | "
            f"%Estructura: {v.get('Pct_estructura') or 'N/D'} | "
            f"Torque máx: {v.get('Torque_max') or 'N/D'}"
        )
        lineas_med.append(f"  Caudal bruto efec: {v.get('Caudal_bruto') or 'N/D'} m³/día")
        lineas_med.append(f"  Carta dinámica [CS]: {m['cs_shape']}")

        if i > 0 and vars_primera:
            campos = [
                ("Carrera_pulg",    "Carrera"),
                ("Golpes_min",      "Golpes/min"),
                ("Diam_piston_pulg","Ø pistón"),
                ("Prof_bomba_m",    "Prof bomba"),
                ("Llenado_pct",     "Llenado %"),
                ("Sumergencia_m",   "Sumergencia"),
                ("Pct_balance",     "%Balance"),
                ("Pct_estructura",  "%Estructura"),
                ("Caudal_bruto",    "Caudal bruto"),
                ("Torque_max",      "Torque máx"),
            ]
            diffs = []
            for key, lbl in campos:
                v0 = _safe_float(vars_primera.get(key))
                v1 = _safe_float(v.get(key))
                if v0 is not None and v1 is not None:
                    delta = round(v1 - v0, 2)
                    sign  = "+" if delta >= 0 else ""
                    diffs.append(f"{lbl}: {v0}→{v1} ({sign}{delta})")
                elif not (v0 is None and v1 is None):
                    diffs.append(f"{lbl}: {v0 or 'N/D'}→{v1 or 'N/D'}")
            if diffs:
                lineas_med.append(f"  ↳ Cambios vs más antigua: {' | '.join(diffs)}")

    if len(mediciones) > 1:
        campos_config = [
            ("Carrera_pulg",    "Carrera"),
            ("Golpes_min",      "Golpes/min"),
            ("Diam_piston_pulg","Ø pistón"),
            ("Prof_bomba_m",    "Prof bomba"),
            ("Tipo_AIB",        "Tipo AIB"),
            ("Potencia_motor",  "Potencia motor"),
        ]
        sin_cambio = []
        for key, lbl in campos_config:
            vals    = [m["vars"].get(key) for m in mediciones]
            vals_ok = [x for x in vals if x is not None]
            if len(vals_ok) == len(mediciones) and all(str(x) == str(vals_ok[0]) for x in vals_ok):
                sin_cambio.append(f"{lbl}={vals_ok[0]}")
        sin_cambio_str = ", ".join(sin_cambio) if sin_cambio else "No determinado"
    else:
        sin_cambio_str = "Solo hay una medición, no aplica comparación temporal."

    n_med = len(mediciones)

    prompt = f"""Eres un ingeniero senior experto en operaciones de pozos petroleros con bombeo mecánico (Rod Pump / Varillado).

Vas a analizar el historial dinamométrico del pozo **{no_key}** y producir un diagnóstico técnico estructurado en JSON.

---
## HISTORIAL DE MEDICIONES ({n_med} DINs, de más antiguo a más reciente)

{"".join(lineas_med)}

---
## VARIABLES SIN CAMBIO entre todas las mediciones
{sin_cambio_str}

---
## INSTRUCCIONES DE ANÁLISIS

### Cómo interpretar la Carta Dinámica [CS]
- **fill_ratio**: >0.55 normal; <0.40 muy delgada → sospecha de gas, fugas o llenado bajo.
- **area**: si cae entre mediciones con misma carrera y golpes/min → pérdida directa de eficiencia.
- **delta_y_subida_inicial**: subida muy brusca → apertura violenta válvula viajera o golpeo de fluido.
- **delta_y_bajada_final**: caída lenta o invertida → posible fuga en válvula fija.
- **pos_carga_max**: pico muy temprano (<20%) puede indicar golpeo; muy tarde (>80%) puede indicar gas.
- **pos_carga_min**: en zona inesperada puede indicar interferencia de fluido.

Compará la evolución entre mediciones: un fill_ratio que cae de 0.62 a 0.41 es diagnóstico directo de deterioro.

### Estados de problemática — solo DOS opciones:
- **ACTIVA**: el problema está presente en el DIN más reciente.
- **RESUELTA**: existía en mediciones anteriores pero en el último DIN ya no está. Si hay una sola medición, todas son ACTIVA.

### Variables sin cambio como clave diagnóstica:
Si Ø pistón, carrera y golpes/min no cambiaron pero el llenado bajó y la sumergencia subió → problema del yacimiento o de la bomba, NO del ajuste operativo. Mencionalo en el resumen.

### Catálogo base (podés agregar nuevas problemáticas si las detectás):
{catalogo_str}

---
## FORMATO DE RESPUESTA — JSON válido, sin texto adicional ni markdown:

{{
  "pozo": "{no_key}",
  "fecha_analisis": "<fecha ISO de hoy>",
  "mediciones_analizadas": {n_med},
  "resumen": "<párrafo de 4-6 oraciones: evolución de la carta, variables que cambiaron, variables estables, conclusión técnica>",
  "problematicas": [
    {{
      "nombre": "<nombre>",
      "severidad": "<BAJA|MEDIA|ALTA|CRÍTICA>",
      "estado": "<ACTIVA|RESUELTA>",
      "descripcion": "<2-3 oraciones: qué evidencia esta problemática y por qué ACTIVA o RESUELTA>"
    }}
  ],
  "variables_sin_cambio": "<lista de variables que no cambiaron, o N/A si hay una sola medición>",
  "recomendacion": "<acción concreta para el próximo paso operativo>",
  "confianza": "<ALTA=3 DINs completos | MEDIA=2 DINs o datos parciales | BAJA=1 DIN o muchos N/D>"
}}
"""
    return prompt


# ------------------------------------------------------------------ #
#  Llamada a OpenAI
# ------------------------------------------------------------------ #

def _call_openai(prompt: str, api_key: str) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=1800,
    )

    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()

    return json.loads(raw)


# ------------------------------------------------------------------ #
#  Generar diagnóstico de un pozo (función base)
# ------------------------------------------------------------------ #

def generar_diagnostico(
    no_key: str,
    din_ok: pd.DataFrame,
    resolve_path_fn,
    gcs_download_fn,
    gcs_bucket: str,
    gcs_prefix: str,
    api_key: str,
) -> dict:
    din_p = din_ok[din_ok["NO_key"] == no_key].copy()
    if din_p.empty or "path" not in din_p.columns:
        return {"error": "Sin archivos DIN disponibles para este pozo."}

    sort_cols = [c for c in ["din_datetime", "mtime"] if c in din_p.columns]
    if sort_cols:
        din_p = din_p.sort_values(sort_cols, na_position="last")
    din_p = din_p.dropna(subset=["path"]).drop_duplicates(subset=["path"]).tail(3)

    mediciones = []
    for _, row in din_p.iterrows():
        path_str = row.get("path")
        if not path_str:
            continue

        p_res = resolve_path_fn(str(path_str))
        if not p_res:
            continue

        local_path = p_res
        if str(p_res).lower().startswith("gs://"):
            try:
                local_path = gcs_download_fn(p_res)
            except Exception:
                continue

        try:
            parsed = _parse_din_full(local_path)
        except Exception:
            continue

        vars_    = _extract_variables(parsed)
        cs_shape = _describe_cs_shape(parsed.get("cs_points", []))

        fecha = (
            row.get("din_datetime")
            or row.get("mtime")
            or vars_.get("FE")
            or "Desconocida"
        )
        if hasattr(fecha, "strftime"):
            fecha = fecha.strftime("%Y-%m-%d %H:%M")

        mediciones.append({
            "fecha":    str(fecha),
            "path":     str(p_res),
            "vars":     vars_,
            "cs_shape": cs_shape,
        })

    if not mediciones:
        return {"error": "No se pudieron parsear archivos DIN para este pozo."}

    prompt = _build_prompt(no_key, mediciones)
    try:
        diag = _call_openai(prompt, api_key)
    except Exception as e:
        return {"error": f"Error llamando a OpenAI: {e}"}

    # Normalizar estados
    for p in diag.get("problematicas", []):
        estado = str(p.get("estado", "")).strip().upper()
        p["estado"] = "RESUELTA" if estado == "RESUELTA" else "ACTIVA"

    diag["_meta"] = {
        "generado_utc":           datetime.now(timezone.utc).isoformat(),
        "paths_analizados":       [m["path"] for m in mediciones],
        "fecha_din_mas_reciente": mediciones[-1]["fecha"] if mediciones else None,
    }

    if gcs_bucket:
        _save_diag_to_gcs(gcs_bucket, no_key, diag, gcs_prefix)

    return diag


# ------------------------------------------------------------------ #
#  Verificar si necesita regenerarse
# ------------------------------------------------------------------ #

def _necesita_regenerar(diag: dict | None, din_ok: pd.DataFrame, no_key: str) -> bool:
    if not diag or "error" in diag:
        return True

    fecha_diag_str = diag.get("_meta", {}).get("generado_utc")
    if not fecha_diag_str:
        return True

    try:
        fecha_diag = pd.to_datetime(fecha_diag_str, utc=True)
    except Exception:
        return True

    din_p = din_ok[din_ok["NO_key"] == no_key].copy()
    if din_p.empty:
        return False

    sort_cols = [c for c in ["din_datetime", "mtime"] if c in din_p.columns]
    if not sort_cols:
        return False

    latest_din = pd.to_datetime(din_p[sort_cols[0]], errors="coerce", utc=True).max()
    if pd.isna(latest_din):
        return False

    return latest_din > fecha_diag


# ------------------------------------------------------------------ #
#  Generación en LOTE — todos los pozos
# ------------------------------------------------------------------ #

def _generar_todos(
    pozos: list[str],
    din_ok: pd.DataFrame,
    resolve_path_fn,
    gcs_download_fn,
    gcs_bucket: str,
    gcs_prefix: str,
    api_key: str,
    solo_pendientes: bool = True,
) -> dict:
    """
    Genera diagnósticos para todos los pozos de la lista.
    Si solo_pendientes=True, saltea los que ya tienen caché vigente.
    Retorna un resumen: { ok: [], error: [], salteados: [] }
    """
    resumen = {"ok": [], "error": [], "salteados": []}

    # Calcular cuáles realmente necesitan generarse
    pozos_a_procesar = []
    for no_key in pozos:
        if solo_pendientes:
            cache = _load_diag_from_gcs(gcs_bucket, no_key, gcs_prefix) if gcs_bucket else None
            if not _necesita_regenerar(cache, din_ok, no_key):
                resumen["salteados"].append(no_key)
                continue
        pozos_a_procesar.append(no_key)

    total = len(pozos_a_procesar)
    if total == 0:
        return resumen

    # UI de progreso
    st.markdown(f"**Generando {total} diagnósticos** ({len(resumen['salteados'])} ya actualizados, salteados)")
    barra       = st.progress(0)
    texto_prog  = st.empty()
    log_area    = st.empty()
    log_lines   = []

    t_inicio = time.time()

    for idx, no_key in enumerate(pozos_a_procesar):
        # Estimar tiempo restante
        elapsed   = time.time() - t_inicio
        velocidad = elapsed / (idx + 0.001)          # seg por pozo
        restantes = total - idx - 1
        eta_seg   = int(velocidad * restantes)
        eta_str   = f"{eta_seg // 60}m {eta_seg % 60}s" if eta_seg >= 60 else f"{eta_seg}s"

        texto_prog.markdown(
            f"⏳ **{no_key}** &nbsp;|&nbsp; "
            f"Pozo {idx + 1} de {total} &nbsp;|&nbsp; "
            f"Tiempo restante estimado: **{eta_str}**"
        )
        barra.progress((idx + 1) / total)

        try:
            diag = generar_diagnostico(
                no_key=no_key,
                din_ok=din_ok,
                resolve_path_fn=resolve_path_fn,
                gcs_download_fn=gcs_download_fn,
                gcs_bucket=gcs_bucket,
                gcs_prefix=gcs_prefix,
                api_key=api_key,
            )
            if "error" in diag:
                resumen["error"].append((no_key, diag["error"]))
                log_lines.append(f"❌ {no_key}: {diag['error']}")
            else:
                resumen["ok"].append(no_key)
                n_prob = len(diag.get("problematicas", []))
                log_lines.append(f"✅ {no_key}: {n_prob} problemática(s)")
        except Exception as e:
            resumen["error"].append((no_key, str(e)))
            log_lines.append(f"❌ {no_key}: {e}")

        # Mostrar últimas 8 líneas del log
        log_area.code("\n".join(log_lines[-8:]), language=None)

    barra.progress(1.0)
    t_total = int(time.time() - t_inicio)
    texto_prog.markdown(
        f"✅ **Listo** — {len(resumen['ok'])} generados, "
        f"{len(resumen['error'])} con error, "
        f"{len(resumen['salteados'])} salteados | "
        f"Tiempo total: {t_total // 60}m {t_total % 60}s"
    )

    return resumen


# ------------------------------------------------------------------ #
#  Render diagnóstico individual
# ------------------------------------------------------------------ #

def _render_diagnostico_individual(diag: dict, no_key: str, bat_map: dict):
    if not diag or "error" in diag:
        st.error(f"Error generando diagnóstico: {diag.get('error', 'desconocido')}")
        return

    bateria         = bat_map.get(no_key, "N/D")
    confianza       = diag.get("confianza", "?")
    mediciones_n    = diag.get("mediciones_analizadas", "?")
    vars_sin_cambio = diag.get("variables_sin_cambio", "N/D")
    problematicas   = diag.get("problematicas", [])

    activas   = sum(1 for p in problematicas if p.get("estado") == "ACTIVA")
    resueltas = sum(1 for p in problematicas if p.get("estado") == "RESUELTA")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Batería",         bateria)
    c2.metric("DINs analizados", mediciones_n)
    c3.metric("Confianza",       confianza)
    c4.metric("⚠️ Activas",      activas)
    c5.metric("✅ Resueltas",    resueltas)

    st.markdown("#### 📝 Resumen ejecutivo")
    st.info(diag.get("resumen", "Sin resumen disponible."))

    st.markdown("#### 🔒 Variables operativas sin cambio entre mediciones")
    st.caption(vars_sin_cambio or "N/D")

    if problematicas:
        st.markdown("#### ⚠️ Problemáticas detectadas")
        st.caption("Ordenadas por severidad. Las RESUELTAS aparecen al final.")

        probs_sorted = sorted(
            problematicas,
            key=lambda x: (
                0 if x.get("estado") == "ACTIVA" else 1,
                SEVERIDAD_ORDEN.get(x.get("severidad", "BAJA"), 9),
            )
        )

        for p in probs_sorted:
            sev          = p.get("severidad", "BAJA")
            estado       = p.get("estado",    "ACTIVA")
            sev_emoji    = SEVERIDAD_EMOJI.get(sev,    "⚪")
            estado_emoji = ESTADO_EMOJI.get(estado,    "")
            sev_color    = SEVERIDAD_COLOR.get(sev,    "#6c757d")
            estado_color = ESTADO_COLOR.get(estado,    "#6c757d")

            st.markdown(
                f"{estado_emoji} {sev_emoji} **{p.get('nombre', '?')}** &nbsp;—&nbsp; "
                f"<span style='color:{sev_color};font-weight:bold'>{sev}</span>"
                f" &nbsp;|&nbsp; "
                f"<span style='color:{estado_color};font-weight:bold'>{estado}</span>",
                unsafe_allow_html=True
            )
            st.caption(p.get("descripcion", ""))
    else:
        st.success("No se detectaron problemáticas.")

    st.markdown("#### 💡 Recomendación")
    st.success(diag.get("recomendacion", "Sin recomendación disponible."))

    with st.expander("Ver JSON completo del diagnóstico"):
        st.json(diag)


# ------------------------------------------------------------------ #
#  Tabla global
# ------------------------------------------------------------------ #

def _build_global_table(diags: dict[str, dict], bat_map: dict, normalize_no_fn) -> pd.DataFrame:
    rows = []
    for no_key, diag in diags.items():
        bateria       = bat_map.get(normalize_no_fn(no_key), "N/D")
        meta          = diag.get("_meta", {})
        fecha_gen     = meta.get("generado_utc", "?")[:19].replace("T", " ")
        confianza     = diag.get("confianza", "?")
        mediciones_n  = diag.get("mediciones_analizadas", "?")
        resumen       = diag.get("resumen", "")
        recomendacion = diag.get("recomendacion", "")

        problematicas = diag.get("problematicas", [])
        if not problematicas:
            rows.append({
                "Pozo":          no_key,
                "Batería":       bateria,
                "Problemática":  "Sin problemáticas",
                "Severidad":     "BAJA",
                "Estado":        "ACTIVA",
                "Descripción":   "",
                "Resumen":       resumen,
                "Recomendación": recomendacion,
                "Mediciones":    mediciones_n,
                "Confianza":     confianza,
                "Generado":      fecha_gen,
            })
        else:
            for p in problematicas:
                rows.append({
                    "Pozo":          no_key,
                    "Batería":       bateria,
                    "Problemática":  p.get("nombre",      "?"),
                    "Severidad":     p.get("severidad",   "BAJA"),
                    "Estado":        p.get("estado",      "ACTIVA"),
                    "Descripción":   p.get("descripcion", ""),
                    "Resumen":       resumen,
                    "Recomendación": recomendacion,
                    "Mediciones":    mediciones_n,
                    "Confianza":     confianza,
                    "Generado":      fecha_gen,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["_est_ord"] = df["Estado"].map({"ACTIVA": 0, "RESUELTA": 1}).fillna(0)
    df["_sev_ord"] = df["Severidad"].map(SEVERIDAD_ORDEN).fillna(9)
    df = df.sort_values(["_est_ord", "_sev_ord", "Batería", "Pozo"]).drop(columns=["_est_ord", "_sev_ord"])
    return df.reset_index(drop=True)


def _render_global_table(df: pd.DataFrame):
    pozos_unicos = df["Pozo"].nunique()
    criticos     = df[(df["Severidad"] == "CRÍTICA") & (df["Estado"] == "ACTIVA")]["Pozo"].nunique()
    altos        = df[(df["Severidad"] == "ALTA")    & (df["Estado"] == "ACTIVA")]["Pozo"].nunique()
    resueltos    = df[df["Estado"] == "RESUELTA"]["Pozo"].nunique()
    sin_prob     = df[df["Problemática"] == "Sin problemáticas"]["Pozo"].nunique()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Pozos diagnosticados",   pozos_unicos)
    k2.metric("🔴 CRÍTICA activa",       criticos)
    k3.metric("🟠 ALTA activa",          altos)
    k4.metric("✅ Con prob. resueltas",  resueltos)
    k5.metric("🟢 Sin problemáticas",    sin_prob)

    st.markdown("#### Filtros")
    f1, f2, f3, f4 = st.columns(4)

    baterias = sorted(df["Batería"].dropna().unique().tolist())
    sevs     = ["CRÍTICA", "ALTA", "MEDIA", "BAJA"]
    estados  = ["ACTIVA", "RESUELTA"]
    probs    = sorted(df["Problemática"].dropna().unique().tolist())

    bat_sel  = f1.multiselect("Batería",      options=baterias, default=baterias,   key="diag_bat_sel")
    sev_sel  = f2.multiselect("Severidad",    options=sevs,     default=sevs,       key="diag_sev_sel")
    est_sel  = f3.multiselect("Estado",       options=estados,  default=["ACTIVA"], key="diag_est_sel")
    prob_sel = f4.multiselect("Problemática", options=probs,    default=probs,      key="diag_prob_sel")

    df_f = df.copy()
    if bat_sel:  df_f = df_f[df_f["Batería"].isin(bat_sel)]
    if sev_sel:  df_f = df_f[df_f["Severidad"].isin(sev_sel)]
    if est_sel:  df_f = df_f[df_f["Estado"].isin(est_sel)]
    if prob_sel: df_f = df_f[df_f["Problemática"].isin(prob_sel)]

    st.caption(f"Mostrando {len(df_f)} filas ({df_f['Pozo'].nunique()} pozos)")
    st.dataframe(df_f, use_container_width=True, height=440, hide_index=True)

    st.markdown("#### 📊 Distribución de problemáticas activas")
    df_chart = df_f[(df_f["Problemática"] != "Sin problemáticas") & (df_f["Estado"] == "ACTIVA")].copy()

    if not df_chart.empty:
        color_sev = {"BAJA": "#28a745", "MEDIA": "#ffc107", "ALTA": "#fd7e14", "CRÍTICA": "#dc3545"}
        g1, g2    = st.columns(2)

        prob_counts = (
            df_chart.groupby(["Problemática", "Severidad"])["Pozo"]
            .nunique().reset_index(name="Pozos")
            .sort_values("Pozos", ascending=True)
        )
        fig1 = px.bar(
            prob_counts, y="Problemática", x="Pozos", color="Severidad",
            color_discrete_map=color_sev, orientation="h",
            title="Pozos afectados por problemática (ACTIVAS)",
            height=max(350, len(prob_counts) * 28)
        )
        g1.plotly_chart(fig1, use_container_width=True)

        bat_sev = df_chart.groupby(["Batería", "Severidad"])["Pozo"].nunique().reset_index(name="Pozos")
        fig2 = px.bar(
            bat_sev, x="Batería", y="Pozos", color="Severidad",
            color_discrete_map=color_sev,
            title="Problemáticas activas por Batería",
            barmode="stack"
        )
        g2.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No hay problemáticas activas para graficar con los filtros actuales.")

    st.markdown("#### ⬇️ Exportar")
    csv_bytes = df_f.to_csv(index=False).encode("utf-8")
    st.download_button("Descargar tabla (CSV)", data=csv_bytes, file_name="diagnosticos_pozos.csv", mime="text/csv")
    try:
        import io
        buf = io.BytesIO()
        df_f.to_excel(buf, index=False, sheet_name="Diagnósticos")
        st.download_button(
            "Descargar tabla (Excel)",
            data=buf.getvalue(),
            file_name="diagnosticos_pozos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception:
        pass


# ------------------------------------------------------------------ #
#  Entry point de la pestaña
# ------------------------------------------------------------------ #

def render_tab_diagnosticos(
    din_ok: pd.DataFrame,
    niv_ok: pd.DataFrame,
    pozo_sel: str,
    parse_din_extras_fn,
    resolve_path_fn,
    gcs_download_fn,
    gcs_bucket: str,
    gcs_prefix: str,
    normalize_no_fn,
    load_coords_fn,
):
    st.subheader("🤖 Diagnósticos IA — Análisis de cartas dinamométricas")

    api_key = _get_openai_key()
    if not api_key:
        st.error(
            "No encontré la API key de OpenAI.\n\n"
            "Configurala en GCP Secret Manager con el nombre **OPENAI_API_KEY** "
            "(o como variable de entorno `OPENAI_API_KEY` para pruebas locales)."
        )
        st.stop()

    if din_ok.empty or "path" not in din_ok.columns:
        st.info("No hay archivos DIN indexados para generar diagnósticos.")
        st.stop()

    # Mapa NO_key → Batería
    coords = load_coords_fn()
    bat_map: dict[str, str] = {}
    if not coords.empty and "nombre_corto" in coords.columns and "nivel_5" in coords.columns:
        for _, row in coords.iterrows():
            k = normalize_no_fn(str(row["nombre_corto"]))
            bat_map[k] = str(row["nivel_5"])

    pozos_con_din = sorted(
        din_ok["NO_key"].dropna().map(normalize_no_fn).loc[lambda s: s != ""].unique().tolist()
    )
    if not pozos_con_din:
        st.info("No hay pozos con DIN disponibles.")
        st.stop()

    # ================================================================
    # BLOQUE: Generación en lote
    # ================================================================
    with st.expander("⚙️ Generación en lote — todos los pozos", expanded=False):

        # Contar estado actual del caché
        if gcs_bucket:
            diags_cache = _load_all_diags_from_gcs(gcs_bucket, pozos_con_din, gcs_prefix)
        else:
            diags_cache = {}

        ya_listos    = len(diags_cache)
        pendientes   = sum(
            1 for pk in pozos_con_din
            if _necesita_regenerar(diags_cache.get(pk), din_ok, pk)
        )
        desactualizados = pendientes - (len(pozos_con_din) - ya_listos)
        desactualizados = max(desactualizados, 0)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total pozos con DIN",     len(pozos_con_din))
        m2.metric("✅ Con diagnóstico",       ya_listos)
        m3.metric("⏳ Pendientes / desact.",  pendientes)

        st.markdown("---")

        col_a, col_b = st.columns(2)
        solo_pend = col_a.checkbox(
            "Saltear pozos ya actualizados",
            value=True,
            help="Si está marcado, solo genera los pozos sin diagnóstico o con DINs nuevos."
        )

        cant_a_generar = pendientes if solo_pend else len(pozos_con_din)
        tiempo_est     = cant_a_generar * 8  # ~8 seg por pozo
        tiempo_str     = f"{tiempo_est // 60}m {tiempo_est % 60}s" if tiempo_est >= 60 else f"{tiempo_est}s"

        col_b.markdown(
            f"**A generar:** {cant_a_generar} pozos &nbsp;|&nbsp; "
            f"**Tiempo estimado:** ~{tiempo_str}"
        )

        if st.button("🚀 Generar todos los diagnósticos", type="primary", use_container_width=True):
            _generar_todos(
                pozos=pozos_con_din,
                din_ok=din_ok,
                resolve_path_fn=resolve_path_fn,
                gcs_download_fn=gcs_download_fn,
                gcs_bucket=gcs_bucket,
                gcs_prefix=gcs_prefix,
                api_key=api_key,
                solo_pendientes=solo_pend,
            )
            st.rerun()

    # ================================================================
    # SECCIÓN A: diagnóstico del pozo seleccionado
    # ================================================================
    st.markdown("---")
    st.markdown(f"### 🔍 Diagnóstico individual — Pozo: **{pozo_sel}**")

    if pozo_sel not in pozos_con_din:
        st.info(f"El pozo **{pozo_sel}** no tiene archivos DIN indexados.")
    else:
        diag_cache = None
        if gcs_bucket:
            with st.spinner("Verificando caché en GCS..."):
                diag_cache = _load_diag_from_gcs(gcs_bucket, pozo_sel, gcs_prefix)

        if _necesita_regenerar(diag_cache, din_ok, pozo_sel):
            msg = "🆕 Hay DINs nuevos — regenerando diagnóstico..." if diag_cache else "📋 Sin diagnóstico previo — generando..."
            with st.spinner(msg):
                diag = generar_diagnostico(
                    no_key=pozo_sel,
                    din_ok=din_ok,
                    resolve_path_fn=resolve_path_fn,
                    gcs_download_fn=gcs_download_fn,
                    gcs_bucket=gcs_bucket,
                    gcs_prefix=gcs_prefix,
                    api_key=api_key,
                )
        else:
            diag    = diag_cache
            meta    = diag.get("_meta", {})
            gen_utc = meta.get("generado_utc", "?")[:19].replace("T", " ")
            din_rec = meta.get("fecha_din_mas_reciente", "?")
            st.caption(f"✅ Caché GCS | Generado: {gen_utc} UTC | DIN más reciente: {din_rec}")

        _render_diagnostico_individual(diag, pozo_sel, bat_map)

    # ================================================================
    # SECCIÓN B: tabla global
    # ================================================================
    st.markdown("---")
    st.markdown("### 📋 Tabla global de problemáticas — todos los pozos")

    if not gcs_bucket:
        st.warning("La vista global requiere GCS (variable DINAS_BUCKET).")
        st.stop()

    with st.spinner("Cargando diagnósticos desde GCS..."):
        diags_globales = _load_all_diags_from_gcs(gcs_bucket, pozos_con_din, gcs_prefix)

    if not diags_globales:
        st.info(
            "Todavía no hay diagnósticos en GCS. "
            "Usá el panel **⚙️ Generación en lote** de arriba para generarlos todos de una vez."
        )
        st.stop()

    df_global = _build_global_table(diags_globales, bat_map, normalize_no_fn)
    if df_global.empty:
        st.info("No hay datos para mostrar.")
        st.stop()

    _render_global_table(df_global)

# ==========================================================
# validaciones_tab.py
# Sistema de validación de sumergencias con historial
#
# Estructura GCS: validaciones/{NO_key}/validaciones.json
# Una entrada por medición (fecha+hora), con historial de cambios.
# ==========================================================

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import streamlit as st


# ------------------------------------------------------------------ #
#  GCS helpers
# ------------------------------------------------------------------ #

def _get_gcs_client():
    try:
        from google.cloud import storage
        return storage.Client()
    except Exception:
        return None


def _blob_name(no_key: str, prefix: str = "") -> str:
    name = f"validaciones/{no_key}/validaciones.json"
    return f"{prefix}/{name}" if prefix else name


def _load_validaciones(bucket_name: str, no_key: str, prefix: str = "") -> dict:
    """Carga el JSON de validaciones de un pozo. Devuelve {} si no existe."""
    client = _get_gcs_client()
    if not client or not bucket_name:
        return {}
    try:
        blob = client.bucket(bucket_name).blob(_blob_name(no_key, prefix))
        if not blob.exists():
            return {}
        return json.loads(blob.download_as_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_validaciones(bucket_name: str, no_key: str, data: dict, prefix: str = "") -> bool:
    client = _get_gcs_client()
    if not client or not bucket_name:
        return False
    try:
        blob = client.bucket(bucket_name).blob(_blob_name(no_key, prefix))
        blob.upload_from_string(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            content_type="application/json"
        )
        return True
    except Exception:
        return False


def _load_all_validaciones(bucket_name: str, pozos: list[str], prefix: str = "") -> dict[str, dict]:
    """Carga validaciones de todos los pozos de una vez."""
    client = _get_gcs_client()
    if not client or not bucket_name:
        return {}
    results = {}
    bucket = client.bucket(bucket_name)
    for no_key in pozos:
        try:
            blob = bucket.blob(_blob_name(no_key, prefix))
            if blob.exists():
                results[no_key] = json.loads(blob.download_as_text(encoding="utf-8"))
        except Exception:
            pass
    return results


# ------------------------------------------------------------------ #
#  Lógica de validación
# ------------------------------------------------------------------ #

def _make_fecha_key(fecha) -> str:
    """Normaliza la fecha a string para usar como key del JSON."""
    if hasattr(fecha, "strftime"):
        return fecha.strftime("%Y-%m-%d %H:%M")
    return str(fecha)[:16]  # recortar a "YYYY-MM-DD HH:MM"


def get_validacion(val_data: dict, fecha_key: str) -> dict:
    """Devuelve el estado de validación de una medición. Default: validada=True."""
    mediciones = val_data.get("mediciones", {})
    if fecha_key in mediciones:
        return mediciones[fecha_key]
    # Default: pre-validada sin comentario
    return {"validada": True, "comentario": "", "historial": []}


def set_validacion(
    val_data: dict,
    no_key: str,
    fecha_key: str,
    validada: bool,
    comentario: str,
    usuario: str,
) -> dict:
    """Actualiza el estado de validación y agrega entrada al historial."""
    if "pozo" not in val_data:
        val_data["pozo"] = no_key
    if "mediciones" not in val_data:
        val_data["mediciones"] = {}

    entrada_actual = val_data["mediciones"].get(fecha_key, {"historial": []})
    historial = entrada_actual.get("historial", [])

    # Solo agregar al historial si algo cambió
    cambio = (
        entrada_actual.get("validada") != validada
        or entrada_actual.get("comentario", "") != comentario
    )
    if cambio:
        historial.append({
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "usuario":   usuario or "anónimo",
            "validada":  validada,
            "comentario": comentario,
        })

    val_data["mediciones"][fecha_key] = {
        "validada":   validada,
        "comentario": comentario,
        "historial":  historial,
    }
    return val_data


# ------------------------------------------------------------------ #
#  Widget de tabla de validaciones (se usa dentro de tab_map)
# ------------------------------------------------------------------ #

def render_tabla_validaciones(
    df_tabla: pd.DataFrame,
    gcs_bucket: str,
    gcs_prefix: str,
    normalize_no_fn,
) -> pd.DataFrame:
    """
    Renderiza la tabla de validación de sumergencias.
    Recibe df_tabla con columnas: NO_key, nivel_5, DT_plot, Sumergencia, Sumergencia_base, etc.
    Devuelve el df con columnas Validada y Comentario añadidas (para exportar).
    """

    # ── Usuario ──────────────────────────────────────────────────────
    if "val_usuario" not in st.session_state:
        st.session_state["val_usuario"] = ""

    col_u, _ = st.columns([2, 5])
    usuario = col_u.text_input(
        "👤 Tu nombre (para el historial)",
        value=st.session_state["val_usuario"],
        key="val_usuario_input",
        placeholder="ej: jperez",
    )
    st.session_state["val_usuario"] = usuario

    # ── Cargar validaciones existentes ───────────────────────────────
    pozos_unicos = df_tabla["NO_key"].dropna().unique().tolist()
    todas_val = _load_all_validaciones(gcs_bucket, pozos_unicos, gcs_prefix)

    # ── Construir filas con estado actual ────────────────────────────
    rows = []
    for _, row in df_tabla.iterrows():
        no_key    = normalize_no_fn(str(row.get("NO_key", "")))
        fecha_raw = row.get("DT_plot")
        fecha_key = _make_fecha_key(fecha_raw)
        sumer     = row.get("Sumergencia")
        base      = row.get("Sumergencia_base", "")

        val_data  = todas_val.get(no_key, {})
        estado    = get_validacion(val_data, fecha_key)

        rows.append({
            "Validada":        estado.get("validada", True),
            "Pozo":            row.get("NO_key", ""),
            "Batería":         row.get("nivel_5", ""),
            "Fecha medición":  fecha_key,
            "Sumer. (m)":      sumer,
            "Base":            base,
            "Comentario":      estado.get("comentario", ""),
            "_no_key":         no_key,
            "_fecha_key":      fecha_key,
        })

    df_edit_base = pd.DataFrame(rows)

    # ── Editor interactivo ───────────────────────────────────────────
    st.caption(
        "✅ = sumergencia validada como real  |  ☐ = marcada como inválida/dudosa  "
        "| Todos los cambios se guardan en GCS con historial."
    )

    col_cfg = {
        "Validada":       st.column_config.CheckboxColumn("✅ Válida", width="small"),
        "Pozo":           st.column_config.TextColumn("Pozo",          width="medium"),
        "Batería":        st.column_config.TextColumn("Batería",        width="small"),
        "Fecha medición": st.column_config.TextColumn("Fecha",          width="medium"),
        "Sumer. (m)":     st.column_config.NumberColumn("Sumer. (m)",   width="small", format="%.1f"),
        "Base":           st.column_config.TextColumn("Base",           width="small"),
        "Comentario":     st.column_config.TextColumn("Comentario",     width="large"),
        # columnas internas ocultas
        "_no_key":        None,
        "_fecha_key":     None,
    }

    edited = st.data_editor(
        df_edit_base,
        column_config=col_cfg,
        use_container_width=True,
        height=420,
        hide_index=True,
        key="val_editor",
    )

    # ── Detectar cambios y guardar ────────────────────────────────────
    cambios = 0
    errores = 0
    for i, edit_row in edited.iterrows():
        orig_row    = df_edit_base.iloc[i]
        no_key      = edit_row["_no_key"]
        fecha_key   = edit_row["_fecha_key"]
        validada    = bool(edit_row["Validada"])
        comentario  = str(edit_row["Comentario"] or "").strip()

        orig_val  = bool(orig_row["Validada"])
        orig_com  = str(orig_row["Comentario"] or "").strip()

        if validada != orig_val or comentario != orig_com:
            val_data = todas_val.get(no_key, {})
            val_data = set_validacion(val_data, no_key, fecha_key, validada, comentario, usuario)
            ok = _save_validaciones(gcs_bucket, no_key, val_data, gcs_prefix)
            if ok:
                todas_val[no_key] = val_data
                cambios += 1
            else:
                errores += 1

    if cambios:
        st.success(f"✅ {cambios} cambio(s) guardado(s) en GCS.")
    if errores:
        st.error(f"❌ {errores} error(es) al guardar. Verificá conexión a GCS.")

    # ── Exportar historial completo ───────────────────────────────────
    st.markdown("#### ⬇️ Exportar historial completo")

    hist_rows = []
    for no_key, val_data in todas_val.items():
        for fecha_key, med in val_data.get("mediciones", {}).items():
            # Estado actual
            hist_rows.append({
                "Pozo":           no_key,
                "Fecha medición": fecha_key,
                "Validada":       med.get("validada", True),
                "Comentario":     med.get("comentario", ""),
                "Tipo":           "ESTADO_ACTUAL",
                "Timestamp":      "",
                "Usuario":        "",
            })
            # Historial de cambios
            for h in med.get("historial", []):
                hist_rows.append({
                    "Pozo":           no_key,
                    "Fecha medición": fecha_key,
                    "Validada":       h.get("validada", True),
                    "Comentario":     h.get("comentario", ""),
                    "Tipo":           "CAMBIO",
                    "Timestamp":      h.get("timestamp", ""),
                    "Usuario":        h.get("usuario", ""),
                })

    if hist_rows:
        df_hist = pd.DataFrame(hist_rows)
        c1, c2 = st.columns(2)
        csv_hist = df_hist.to_csv(index=False).encode("utf-8")
        c1.download_button(
            "📄 Descargar historial (CSV)",
            data=csv_hist,
            file_name="historial_validaciones.csv",
            mime="text/csv",
        )
        try:
            import io
            buf = io.BytesIO()
            df_hist.to_excel(buf, index=False, sheet_name="Historial")
            c2.download_button(
                "📊 Descargar historial (Excel)",
                data=buf.getvalue(),
                file_name="historial_validaciones.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception:
            pass
    else:
        st.info("Todavía no hay validaciones guardadas.")

    # Devolver df con columnas de validación para uso externo (filtro del mapa)
    edited_export = edited[["Pozo", "Batería", "Fecha medición", "Sumer. (m)", "Base", "Validada", "Comentario"]].copy()
    return edited_export


# ------------------------------------------------------------------ #
#  Helper para filtrar snap_map según validaciones
# ------------------------------------------------------------------ #

def filtrar_por_validacion(
    snap_map: pd.DataFrame,
    gcs_bucket: str,
    gcs_prefix: str,
    normalize_no_fn,
    solo_validadas: bool,
) -> pd.DataFrame:
    """
    Filtra snap_map por estado de validación.
    solo_validadas=True  → solo las validadas (sumergencias reales del campo)
    solo_validadas=False → solo las NO validadas (marcadas como dudosas)
    """
    if not gcs_bucket:
        return snap_map

    pozos = snap_map["NO_key"].dropna().unique().tolist()
    todas_val = _load_all_validaciones(gcs_bucket, pozos, gcs_prefix)

    def es_valida(row):
        no_key    = normalize_no_fn(str(row.get("NO_key", "")))
        fecha_key = _make_fecha_key(row.get("DT_plot"))
        val_data  = todas_val.get(no_key, {})
        estado    = get_validacion(val_data, fecha_key)
        return estado.get("validada", True)  # default: válida

    mask = snap_map.apply(es_valida, axis=1)
    return snap_map[mask].copy() if solo_validadas else snap_map[~mask].copy()

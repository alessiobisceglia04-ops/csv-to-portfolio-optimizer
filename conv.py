from io import BytesIO
 
import numpy as np
import pandas as pd
import streamlit as st
 
 
# =========================================================
# CONFIGURAZIONE PAGINA
# =========================================================
 
st.set_page_config(
    page_title="CSV Market Data Cleaner",
    page_icon="📊",
    layout="wide",
)
 
 
# =========================================================
# LETTURA CSV
# =========================================================
 
def read_csv_robustly(uploaded_file) -> pd.DataFrame:
    """
    Prova a leggere il CSV con diversi separatori
    e diversi formati decimali.
    """
 
    file_content = uploaded_file.getvalue()
 
    attempts = [
        {"sep": ",", "decimal": "."},
        {"sep": ";", "decimal": ","},
        {"sep": ";", "decimal": "."},
        {"sep": ",", "decimal": ","},
    ]
 
    for settings in attempts:
        try:
            dataframe = pd.read_csv(
                BytesIO(file_content),
                sep=settings["sep"],
                decimal=settings["decimal"],
            )
 
            if dataframe.shape[1] >= 2:
                return dataframe
 
        except Exception:
            continue
 
    raise ValueError(
        "Non riesco a leggere correttamente il CSV. "
        "Controlla che contenga una colonna Date "
        "e almeno una colonna di prezzi."
    )
 
 
# =========================================================
# CONVERSIONE PREZZI
# =========================================================
 
def convert_price_column(series: pd.Series) -> pd.Series:
    """
    Converte una colonna di prezzi in valori numerici.
 
    Gestisce formati come:
    71,85 €
    2.273,75 €
    2273.75
    2,273.75
    None
    NaN
    """
 
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
 
    cleaned = series.astype(str).str.strip()
 
    for symbol in ["€", "$", "£"]:
        cleaned = cleaned.str.replace(symbol, "", regex=False)
 
    cleaned = cleaned.str.replace("\u00a0", "", regex=False)
    cleaned = cleaned.str.replace(" ", "", regex=False)
 
    missing_values = {
        "",
        "nan",
        "none",
        "null",
        "-",
        "--",
        "n/a",
        "na",
    }
 
    def normalize_number(value: str) -> str:
        value = value.strip()
 
        if value.lower() in missing_values:
            return ""
 
        # Formato europeo: 2.273,75
        if "." in value and "," in value:
            if value.rfind(",") > value.rfind("."):
                value = value.replace(".", "")
                value = value.replace(",", ".")
 
            # Formato inglese: 2,273.75
            else:
                value = value.replace(",", "")
 
        # Formato con sola virgola decimale: 71,85
        elif "," in value:
            value = value.replace(",", ".")
 
        return value
 
    cleaned = cleaned.apply(normalize_number)
 
    return pd.to_numeric(cleaned, errors="coerce")
 
 
# =========================================================
# ELABORAZIONE DATI
# =========================================================
 
def process_data(raw_data: pd.DataFrame, target_frequency: str = "Auto"):
    """
    Pulisce i prezzi, li allinea su una griglia temporale regolare
    (giornaliera, settimanale o mensile) e calcola i log return
    sulla griglia, così ogni rendimento copre lo stesso orizzonte.
    """
 
    if raw_data.empty:
        raise ValueError("Il CSV è vuoto.")
 
    if raw_data.shape[1] < 2:
        raise ValueError(
            "Il CSV deve contenere almeno due colonne: "
            "Date e almeno un asset."
        )
 
    initial_rows = len(raw_data)
 
    data = raw_data.copy()
 
    # La prima colonna viene considerata la colonna delle date.
    original_date_column = data.columns[0]
    data = data.rename(columns={original_date_column: "Date"})
 
    # Converte le date.
    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce",
        dayfirst=True,
    )
 
    invalid_dates = int(data["Date"].isna().sum())
 
    # Elimina le date non riconosciute.
    data = data.dropna(subset=["Date"])
 
    asset_columns = [
        column
        for column in data.columns
        if column != "Date"
    ]
 
    if not asset_columns:
        raise ValueError(
            "Non sono state trovate colonne contenenti prezzi."
        )
 
    # Converte tutte le colonne dei prezzi in numeriche.
    for column in asset_columns:
        data[column] = convert_price_column(data[column])
 
    # Elimina date duplicate mantenendo l'ultima osservazione.
    duplicate_dates = int(
        data["Date"].duplicated(keep="last").sum()
    )
 
    data = data.drop_duplicates(
        subset="Date",
        keep="last",
    )
 
    # Ordina le date dalla più vecchia alla più recente.
    data = data.sort_values("Date").reset_index(drop=True)
 
    # =====================================================
    # REPORT DIAGNOSTICO
    # =====================================================
 
    diagnostic_rows = []
 
    for column in asset_columns:
        valid_mask = data[column].notna()
 
        valid_values = int(valid_mask.sum())
        missing_values = int(data[column].isna().sum())
 
        if valid_values > 0:
            first_valid_date = data.loc[
                valid_mask,
                "Date",
            ].min()
 
            last_valid_date = data.loc[
                valid_mask,
                "Date",
            ].max()
 
        else:
            first_valid_date = pd.NaT
            last_valid_date = pd.NaT
 
        diagnostic_rows.append(
            {
                "Asset": column,
                "Valid values": valid_values,
                "Missing values": missing_values,
                "First valid date": first_valid_date,
                "Last valid date": last_valid_date,
            }
        )
 
    diagnostic_report = pd.DataFrame(diagnostic_rows)
 
    # Controlla se esistono asset completamente vuoti.
    empty_assets = [
        column
        for column in asset_columns
        if data[column].notna().sum() == 0
    ]
 
    if empty_assets:
        raise ValueError(
            "Questi asset non contengono alcun prezzo valido: "
            + ", ".join(empty_assets)
        )
 
    rows_before_cleaning = len(data)
 
    # =====================================================
    # ALLINEAMENTO PREZZI SU GRIGLIA REGOLARE
    # =====================================================
    #
    # Eliminare l'intera riga quando manca anche un solo prezzo
    # e POI calcolare i rendimenti produce rendimenti che coprono
    # orizzonti diversi (2 giorni, una settimana, un mese...):
    # covarianza e annualizzazione perdono significato.
    #
    # Approccio corretto:
    #   1. si stima la frequenza nativa di ogni asset
    #      (distanza mediana tra le sue osservazioni valide);
    #   2. si sceglie la griglia comune (la più fitta sostenibile
    #      da TUTTI gli asset, oppure quella scelta dall'utente);
    #   3. si portano i prezzi sulla griglia (ultimo prezzo
    #      disponibile nel periodo, con forward-fill limitato
    #      per i buchi brevi tipo festività);
    #   4. SOLO ORA si calcolano i log return, tutti sullo
    #      stesso orizzonte.
 
    prices = data.set_index("Date")[asset_columns].copy()
 
    # Frequenza nativa di ciascun asset. Si usa il 90° percentile dei gap,
    # non la mediana: se le osservazioni sono raggruppate (es. pochi giorni
    # consecutivi al mese), la mediana vede gap di 1 giorno e ignora i buchi
    # ricorrenti; il percentile alto cattura il buco tipico più ampio,
    # cioè la griglia più fitta su cui l'asset ha davvero un dato per periodo.
    native_gaps = {}
    for column in asset_columns:
        valid_dates = prices[column].dropna().index.to_series().sort_values()
        if len(valid_dates) >= 3:
            gaps_asset = valid_dates.diff().dt.days.dropna()
            native_gaps[column] = float(gaps_asset.quantile(0.90))
        else:
            native_gaps[column] = np.nan
 
    diagnostic_report["Typical gap p90 (days)"] = [
        native_gaps[column] for column in diagnostic_report["Asset"]
    ]
 
    worst_gap = float(np.nanmax(list(native_gaps.values())))
 
    def _grid_for(gap_days: float):
        if gap_days <= 6:
            return "B", "giornaliera", 5
        if gap_days <= 12:
            return "W-FRI", "settimanale", 2
        return "ME", "mensile", 1
 
    frequency_labels = {
        "Giornaliera": 3.0,
        "Settimanale": 7.0,
        "Mensile": 30.0,
    }
 
    if target_frequency in frequency_labels:
        rule, freq_name, ffill_limit = _grid_for(
            frequency_labels[target_frequency]
        )
        forced = True
    else:
        rule, freq_name, ffill_limit = _grid_for(worst_gap)
        forced = False
 
    frequency_warning = None
    if forced and frequency_labels[target_frequency] < worst_gap / 2:
        frequency_warning = (
            f"Hai forzato la frequenza {target_frequency.lower()}, ma almeno un asset "
            f"ha osservazioni ogni ~{worst_gap:.0f} giorni: i suoi prezzi verranno "
            f"riportati con forward-fill e la sua volatilità risulterà sottostimata. "
            f"Considera una frequenza più bassa."
        )
 
    # Griglia regolare: ultimo prezzo disponibile in ogni periodo,
    # con forward-fill limitato per coprire festività e buchi brevi.
    resampled_prices = (
        prices.resample(rule).last().ffill(limit=ffill_limit)
    )
 
    filled_cells = int(
        resampled_prices.notna().sum().sum()
        - prices.resample(rule).last().notna().sum().sum()
    )
 
    # Elimina i periodi iniziali/residui in cui un asset non esiste ancora.
    clean_prices = resampled_prices.dropna(how="any").reset_index()
 
    removed_missing_rows = rows_before_cleaning - len(clean_prices)
 
    if len(clean_prices) < 2:
        raise ValueError(
            "Dopo l'allineamento sulla griglia comune sono rimaste "
            f"solo {len(clean_prices)} righe complete. "
            "Servono almeno due periodi comuni a tutti gli asset."
        )
 
    # Controlla prezzi uguali o inferiori a zero.
    non_positive = clean_prices[asset_columns] <= 0
 
    if non_positive.any().any():
        invalid_assets = (
            non_positive.any()
            .loc[lambda values: values]
            .index
            .tolist()
        )
 
        raise ValueError(
            "Sono presenti prezzi uguali o inferiori a zero negli asset: "
            + ", ".join(invalid_assets)
        )
 
    # =====================================================
    # CALCOLO LOG RETURN (sulla griglia regolare)
    # =====================================================
 
    # Formula:
    # ln(Pt / Pt-1)
    #
    # Il risultato viene mantenuto in forma decimale.
    # Esempio:
    # -0.0385 verrà mostrato in Excel come -3.85%.
    log_returns_values = np.log(
        clean_prices[asset_columns]
        / clean_prices[asset_columns].shift(1)
    )
 
    log_returns = pd.concat(
        [
            clean_prices[["Date"]],
            log_returns_values,
        ],
        axis=1,
    )
 
    # Elimina la prima riga, perché non ha un prezzo precedente.
    log_returns = log_returns.dropna(
        subset=asset_columns,
        how="any",
    ).reset_index(drop=True)
 
    clean_prices = clean_prices.reset_index(drop=True)
 
    if log_returns.empty:
        raise ValueError(
            "Non è stato possibile calcolare i log return."
        )
 
    report = {
        "initial_rows": initial_rows,
        "invalid_dates": invalid_dates,
        "duplicate_dates": duplicate_dates,
        "removed_missing_rows": removed_missing_rows,
        "clean_price_rows": len(clean_prices),
        "log_return_rows": len(log_returns),
        "assets": len(asset_columns),
        "first_date": clean_prices["Date"].min(),
        "last_date": clean_prices["Date"].max(),
        "frequency": freq_name,
        "worst_native_gap": worst_gap,
        "filled_cells": filled_cells,
        "frequency_warning": frequency_warning,
    }
 
    return (
        clean_prices,
        log_returns,
        diagnostic_report,
        report,
    )
 
 
# =========================================================
# CREAZIONE FILE EXCEL
# =========================================================
 
def create_excel_file(
    clean_prices: pd.DataFrame,
    log_returns: pd.DataFrame,
) -> bytes:
    """
    Crea un file Excel in memoria con due fogli:
    Clean Prices e Log Returns.
    """
 
    output = BytesIO()
 
    with pd.ExcelWriter(
        output,
        engine="openpyxl",
        datetime_format="DD/MM/YYYY",
    ) as writer:
 
        clean_prices.to_excel(
            writer,
            sheet_name="Clean Prices",
            index=False,
        )
 
        log_returns.to_excel(
            writer,
            sheet_name="Log Returns",
            index=False,
        )
 
        clean_sheet = writer.sheets["Clean Prices"]
        returns_sheet = writer.sheets["Log Returns"]
 
        # Blocca la prima riga e la colonna Date.
        clean_sheet.freeze_panes = "B2"
        returns_sheet.freeze_panes = "B2"
 
        # Aggiunge i filtri.
        clean_sheet.auto_filter.ref = clean_sheet.dimensions
        returns_sheet.auto_filter.ref = returns_sheet.dimensions
 
        # Larghezza colonna Date.
        clean_sheet.column_dimensions["A"].width = 14
        returns_sheet.column_dimensions["A"].width = 14
 
        asset_columns = [
            column
            for column in clean_prices.columns
            if column != "Date"
        ]
 
        # Regola la larghezza delle colonne.
        for column_index, asset_name in enumerate(
            asset_columns,
            start=2,
        ):
            column_letter = clean_sheet.cell(
                row=1,
                column=column_index,
            ).column_letter
 
            width = max(
                len(str(asset_name)) + 3,
                14,
            )
 
            clean_sheet.column_dimensions[
                column_letter
            ].width = width
 
            returns_sheet.column_dimensions[
                column_letter
            ].width = width
 
        # Formato date.
        for cell in clean_sheet["A"][1:]:
            cell.number_format = "DD/MM/YYYY"
 
        for cell in returns_sheet["A"][1:]:
            cell.number_format = "DD/MM/YYYY"
 
        # Formato prezzi.
        for row in clean_sheet.iter_rows(
            min_row=2,
            min_col=2,
        ):
            for cell in row:
                cell.number_format = "0.0000"
 
        # Formato percentuale dei log return.
        #
        # Il valore nella cella resta decimale:
        # -0.03858557
        #
        # Excel lo mostra come:
        # -3.858557%
        for row in returns_sheet.iter_rows(
            min_row=2,
            min_col=2,
        ):
            for cell in row:
                cell.number_format = "0.000000%"
 
    output.seek(0)
 
    return output.getvalue()
 
 
# =========================================================
# INTERFACCIA STREAMLIT
# =========================================================
 
_ICO_RAIL = {
    "logo":  "<path d='M3 17l5-6 4 4 6-8 3 4'/>",
    "home":  "<path d='M3 10.5 12 3l9 7.5'/><path d='M5 9.5V21h14V9.5'/>",
    "chart": "<path d='M3 3v18h18'/><rect x='7' y='11' width='3' height='7'/><rect x='12' y='7' width='3' height='11'/><rect x='17' y='13' width='3' height='5'/>",
    "table": "<rect x='3' y='4' width='18' height='16' rx='2'/><path d='M3 10h18'/><path d='M9 10v10'/>",
    "file":  "<path d='M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z'/><path d='M14 2v6h6'/><path d='M12 12v6'/><path d='m9 15 3 3 3-3'/>",
    "gear":  "<circle cx='12' cy='12' r='3'/><path d='M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1-1.6 1.7 1.7 0 0 0-1.9.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.6-1 1.7 1.7 0 0 0-.3-1.9l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.9.3h0a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5h0a1.7 1.7 0 0 0 1.9-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.9v0a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z'/>",
    "cloud": "<path d='M4 14.9A7 7 0 1 1 15.7 8h1.8a4.5 4.5 0 0 1 2.5 8.2'/><path d='M12 12v9'/><path d='m8 17 4-4 4 4'/>",
    "cal":   "<rect x='3' y='4' width='18' height='18' rx='2'/><path d='M16 2v4'/><path d='M8 2v4'/><path d='M3 10h18'/>",
    "shield":"<path d='M12 22s8-3 8-10V5l-8-3-8 3v7c0 7 8 10 8 10z'/><path d='m9 12 2 2 4-4'/>",
    "pct":   "<path d='m19 5-14 14'/><circle cx='6.5' cy='6.5' r='2.5'/><circle cx='17.5' cy='17.5' r='2.5'/>",
    "down":  "<path d='M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'/><path d='m7 10 5 5 5-5'/><path d='M12 15V3'/>",
    "list":  "<path d='M8 6h13'/><path d='M8 12h13'/><path d='M8 18h13'/><path d='M3 6h.01'/><path d='M3 12h.01'/><path d='M3 18h.01'/>",
    "check": "<circle cx='12' cy='12' r='10'/><path d='m9 12 2 2 4-4'/>",
}
 
 
def _svg_uri(body, color="%234C7DFF"):
    return (
        "url(\"data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' "
        f"viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='2' "
        f"stroke-linecap='round' stroke-linejoin='round'>{body}</svg>\")"
    )
 
 
CSS_CONV = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
 
#MainMenu, footer, header,
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ display: none !important; }}
 
:root {{
    --bg: #050505; --panel: #0D0D0F; --panel-2: #141417;
    --border: #24242A; --text: #F7F7F8; --text-2: #9A9AA3;
    --accent: #4C7DFF; --accent-h: #6B93FF;
    --success: #22C55E; --danger: #F87171;
}}
 
html, body, [class*="css"] {{ font-family: 'Inter', -apple-system, sans-serif; }}
.stApp {{ background: var(--bg); }}
h1, h2, h3, h4 {{ color: var(--text) !important; }}
hr {{ border-color: var(--border) !important; }}
*:focus-visible {{ outline: 2px solid var(--accent-h); outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation: none !important; transition: none !important; }} }}
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: var(--panel); }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
 
/* contenuto spostato a destra per fare spazio al rail */
[data-testid="stAppViewContainer"] .block-container {{
    padding-left: 104px; padding-top: 2.2rem; max-width: 1280px;
}}
@media (max-width: 740px) {{
    .rail {{ display: none; }}
    [data-testid="stAppViewContainer"] .block-container {{ padding-left: 1rem; }}
}}
 
/* ── Rail laterale ── */
.rail {{
    position: fixed; left: 14px; top: 14px; bottom: 14px; width: 60px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 16px; z-index: 998;
    display: flex; flex-direction: column; align-items: center;
    padding: 12px 0; gap: 8px;
}}
.rail .logo {{
    width: 36px; height: 36px; border-radius: 10px;
    background-image: {_svg_uri(_ICO_RAIL['logo'])};
    background-repeat: no-repeat; background-position: center; background-size: 24px;
    margin-bottom: 14px;
}}
.rail a {{
    width: 40px; height: 40px; border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    background-repeat: no-repeat; background-position: center; background-size: 19px;
    transition: background-color .15s ease;
}}
.rail a:hover {{ background-color: var(--panel-2); }}
.rail a.on {{ background-color: rgba(76,125,255,0.16); border: 1px solid rgba(76,125,255,0.5); }}
.rail .sp {{ flex: 1; }}
 
/* ── Header ── */
.cv-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 14px; }}
.cv-title {{ font-size: 2.15rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1.15; color: var(--text); }}
.cv-title b {{ color: var(--accent); font-weight: 800; }}
.cv-sub {{ color: var(--text-2); font-size: 0.95rem; margin-top: 8px; max-width: 560px; }}
 
/* ── Drop zone (restyling ADDITIVO: nessun elemento nascosto) ── */
[data-testid="stFileUploader"] {{ background: transparent; border: none; padding: 0; }}
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploader"] section {{
    background: rgba(13,13,15,0.7) !important;
    border: 1.5px dashed rgba(76,125,255,0.5) !important;
    border-radius: 14px !important;
    padding: 30px 26px !important;
    transition: border-color .2s ease, background .2s ease;
    cursor: pointer;
}}
[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploader"] section:hover {{
    border-color: var(--accent-h) !important;
    background: var(--panel-2) !important;
}}
[data-testid="stFileUploaderDropzone"] svg {{
    width: 62px !important; height: 62px !important;
    padding: 16px; box-sizing: border-box;
    background: rgba(76,125,255,0.10); border-radius: 50%;
    color: var(--accent) !important; fill: var(--accent) !important;
    margin-right: 8px;
}}
[data-testid="stFileUploaderDropzoneInstructions"] span {{
    font-size: 1.08rem !important; font-weight: 700 !important; color: var(--text) !important;
}}
[data-testid="stFileUploaderDropzoneInstructions"] small {{
    font-size: 0.83rem !important; color: var(--text-2) !important;
}}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] section button {{
    background: var(--accent) !important; color: #fff !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; padding: 0.62rem 1.5rem !important;
    transition: background .15s ease;
}}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploader"] section button:hover {{ background: var(--accent-h) !important; }}
[data-testid="stFileUploaderFile"] {{ color: var(--text) !important; }}
[data-testid="stFileUploaderFile"] small {{ color: var(--text-2) !important; }}
 
/* ── Striscia feature ── */
.feat-strip {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    gap: 8px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 14px; padding: 18px 20px; margin-top: 14px;
}}
.fs {{ display: flex; gap: 12px; align-items: flex-start; }}
.fs-ico {{
    width: 38px; height: 38px; flex-shrink: 0; border-radius: 10px;
    background-color: rgba(76,125,255,0.10);
    background-repeat: no-repeat; background-position: center; background-size: 19px;
}}
.fs .t {{ color: var(--accent); font-weight: 700; font-size: 0.9rem; }}
.fs .s {{ color: var(--text-2); font-size: 0.78rem; margin-top: 2px; line-height: 1.45; }}
 
/* ── Card file caricato ── */
.file-ok {{
    display: flex; align-items: center; gap: 14px;
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 14px; padding: 15px 18px; margin-top: 14px;
}}
.file-ok .doc {{
    width: 42px; height: 46px; border-radius: 8px; flex-shrink: 0;
    background: var(--success);
    display: flex; align-items: flex-end; justify-content: center;
    color: #fff; font-size: 0.6rem; font-weight: 800; padding-bottom: 5px;
}}
.file-ok .n {{ color: var(--text); font-weight: 600; font-size: 0.95rem; }}
.file-ok .m {{ color: var(--text-2); font-size: 0.8rem; margin-top: 2px; }}
.file-ok .ok {{ color: var(--success); font-weight: 600; }}
 
/* ── Card statistiche ── */
.stat-row {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 10px; margin-top: 14px;
}}
.stat {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 14px; padding: 18px 16px;
    transition: border-color .2s ease;
}}
.stat:hover {{ border-color: var(--accent); }}
.stat .ico {{
    width: 34px; height: 34px; border-radius: 9px; margin-bottom: 12px;
    background-color: rgba(76,125,255,0.10);
    background-repeat: no-repeat; background-position: center; background-size: 17px;
}}
.stat .v {{ color: var(--text); font-weight: 700; font-size: 1.35rem; font-variant-numeric: tabular-nums; line-height: 1.2; }}
.stat .v.sm {{ font-size: 0.98rem; }}
.stat .l {{ color: var(--text); font-size: 0.82rem; margin-top: 6px; }}
.stat .s {{ color: var(--text-2); font-size: 0.72rem; margin-top: 1px; }}
 
/* ── Bottoni ── */
div.stButton > button {{
    background: var(--accent); color: #fff !important;
    border: none !important; border-radius: 12px;
    padding: 0.8rem 2rem; font-size: 1rem; font-weight: 600;
    width: 100%; transition: background .15s ease;
}}
div.stButton > button:hover:not(:disabled) {{ background: var(--accent-h); }}
[data-testid="stDownloadButton"] > button {{
    background: var(--accent) !important; color: #fff !important;
    border: none !important; border-radius: 12px !important;
    font-weight: 600 !important; padding: 0.75rem 2rem !important;
    width: 100% !important; transition: background .15s ease !important;
}}
[data-testid="stDownloadButton"] > button:hover {{ background: var(--accent-h) !important; }}
 
/* ── Selectbox / tabs / expander / metric / dataframe ── */
[data-testid="stSelectbox"] > div > div {{
    background: var(--panel-2) !important; border: 1px solid var(--border) !important;
    border-radius: 10px !important; color: var(--text) !important;
}}
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 4px; gap: 4px;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    background: transparent !important; color: var(--text-2) !important;
    border-radius: 8px; padding: 8px 20px; font-weight: 500; border: none !important;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    background: rgba(76,125,255,0.12) !important; color: var(--accent) !important;
}}
[data-testid="stMetric"] {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 14px; padding: 14px 16px;
}}
[data-testid="stMetricLabel"] {{ color: var(--text-2) !important; }}
[data-testid="stMetricValue"] {{ color: var(--text) !important; }}
.sec-label {{
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-2);
    margin: 26px 0 12px; padding-bottom: 10px;
    border-bottom: 1px solid var(--border);
}}
 
/* ── Footer sicurezza ── */
.sec-note {{
    text-align: center; color: var(--text-2); font-size: 0.8rem;
    margin-top: 14px;
}}
</style>
"""
 
st.markdown(CSS_CONV, unsafe_allow_html=True)
 
# ── Rail laterale (decorativo, con ancore alle sezioni) ─────────────────────
st.markdown(
    '<div class="rail">'
    '<div class="logo"></div>'
    f'<a class="on" href="#top" style="background-image:{_svg_uri(_ICO_RAIL["home"])};"></a>'
    f'<a href="#metriche" style="background-image:{_svg_uri(_ICO_RAIL["chart"], "%239A9AA3")};"></a>'
    f'<a href="#tabelle" style="background-image:{_svg_uri(_ICO_RAIL["table"], "%239A9AA3")};"></a>'
    f'<a href="#download" style="background-image:{_svg_uri(_ICO_RAIL["file"], "%239A9AA3")};"></a>'
    '<div class="sp"></div>'
    f'<a href="#top" style="background-image:{_svg_uri(_ICO_RAIL["gear"], "%239A9AA3")};"></a>'
    '</div>',
    unsafe_allow_html=True,
)
 
# ── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div id="top"></div>', unsafe_allow_html=True)
head_l, head_r = st.columns([5, 1])
with head_l:
    st.markdown(
        '<div class="cv-title">CSV <b>Market Data Cleaner</b></div>'
        '<div class="cv-sub">Pulisci i tuoi dati finanziari, allinea le serie temporali '
        'e calcola i log return in modo automatico e affidabile.</div>',
        unsafe_allow_html=True,
    )
with head_r:
    with st.popover("❔ Guida", use_container_width=True):
        st.markdown(
            """
**Come si usa**
 
1. Carica un CSV con la **data nella prima colonna** e un asset
   (prezzi di chiusura) per ogni colonna successiva.
2. Scegli la frequenza dei rendimenti — **Auto** è consigliato:
   allinea tutti gli asset sulla griglia più fitta che i dati reggono.
3. Premi **Processa il file** e scarica l'Excel con prezzi puliti
   e log return.
 
I log return sono calcolati **dopo** l'allineamento temporale,
così ogni rendimento copre lo stesso orizzonte.
            """
        )
 
st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
 
# ── Upload ───────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Carica il tuo file CSV — trascina e rilascia il file qui o sfoglia",
    type=["csv"],
)
 
# ── Striscia feature ─────────────────────────────────────────────────────────
st.markdown(
    '<div class="feat-strip">'
    f'<div class="fs"><div class="fs-ico" style="background-image:{_svg_uri(_ICO_RAIL["cal"])};"></div>'
    '<div><div class="t">Allineamento automatico</div>'
    '<div class="s">Allinea i prezzi su una griglia temporale regolare</div></div></div>'
    f'<div class="fs"><div class="fs-ico" style="background-image:{_svg_uri(_ICO_RAIL["shield"])};"></div>'
    '<div><div class="t">Pulizia dei dati</div>'
    '<div class="s">Rimuove righe incomplete e dati non validi</div></div></div>'
    f'<div class="fs"><div class="fs-ico" style="background-image:{_svg_uri(_ICO_RAIL["pct"])};"></div>'
    '<div><div class="t">Log return</div>'
    '<div class="s">Calcola i rendimenti logaritmici in percentuale</div></div></div>'
    f'<div class="fs"><div class="fs-ico" style="background-image:{_svg_uri(_ICO_RAIL["down"])};"></div>'
    '<div><div class="t">Export Excel</div>'
    '<div class="s">Esporta i dati puliti e i log return in un file Excel pronto all\'uso</div></div></div>'
    '</div>',
    unsafe_allow_html=True,
)
 
if uploaded_file is not None:
 
    try:
        raw_data = read_csv_robustly(uploaded_file)
 
        # ── Card file + statistiche pre-elaborazione ─────────────────────────
        size_kb = len(uploaded_file.getvalue()) / 1024
        size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.2f} MB"
 
        st.markdown(
            f'<div class="file-ok">'
            f'<div class="doc">CSV</div>'
            f'<div><div class="n">{uploaded_file.name}</div>'
            f'<div class="m">{size_str} &nbsp;•&nbsp; <span class="ok">Caricato con successo</span></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
 
        n_righe   = len(raw_data)
        colonne   = list(raw_data.columns)
        n_asset   = max(len(colonne) - 1, 0)
        complete  = int(raw_data.notna().all(axis=1).sum())
        incomplete = n_righe - complete
        date_prova = pd.to_datetime(raw_data[colonne[0]], errors="coerce", dayfirst=True)
        if date_prova.notna().any():
            intervallo = (
                f"{date_prova.min().strftime('%d/%m/%Y')}<br>{date_prova.max().strftime('%d/%m/%Y')}"
            )
        else:
            intervallo = "—"
 
        st.markdown(
            '<div class="stat-row">'
            f'<div class="stat"><div class="ico" style="background-image:{_svg_uri(_ICO_RAIL["list"])};"></div>'
            f'<div class="v">{n_righe:,}</div><div class="l">Righe totali</div><div class="s">Nel file originale</div></div>'.replace(",", ".")
            + f'<div class="stat"><div class="ico" style="background-image:{_svg_uri(_ICO_RAIL["list"])};"></div>'
            f'<div class="v">{complete:,}</div><div class="l">Righe complete</div><div class="s">Con tutti i prezzi presenti</div></div>'.replace(",", ".")
            + f'<div class="stat"><div class="ico" style="background-image:{_svg_uri(_ICO_RAIL["chart"])};"></div>'
            f'<div class="v">{n_asset}</div><div class="l">Asset individuati</div><div class="s">Colonne di prezzo</div></div>'
            + f'<div class="stat"><div class="ico" style="background-image:{_svg_uri(_ICO_RAIL["cal"])};"></div>'
            f'<div class="v sm">{intervallo}</div><div class="l">Intervallo temporale</div><div class="s">Primo e ultimo giorno</div></div>'
            + f'<div class="stat"><div class="ico" style="background-image:{_svg_uri(_ICO_RAIL["check"])};"></div>'
            f'<div class="v">{incomplete:,}</div><div class="l">Righe incomplete</div><div class="s">Con valori mancanti, verranno allineate</div></div>'.replace(",", ".")
            + '</div>',
            unsafe_allow_html=True,
        )
 
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
 
        sel_col, _ = st.columns([1, 2])
        with sel_col:
            target_frequency = st.selectbox(
                "Frequenza dei rendimenti",
                ["Auto (consigliato)", "Giornaliera", "Settimanale", "Mensile"],
                help=(
                    "Auto sceglie la griglia più fitta sostenibile da tutti gli asset: "
                    "se un asset ha dati solo settimanali o mensili, l'intera serie "
                    "viene portata a quella frequenza per mantenere coerente il calcolo "
                    "di volatilità e correlazioni."
                ),
            )
        target_frequency = target_frequency.split(" ")[0]
 
        if st.button("✨ Processa il file", type="primary", use_container_width=True):
            with st.spinner("Pulizia dei dati e calcolo dei log return..."):
                (
                    clean_prices,
                    log_returns,
                    diagnostic_report,
                    report,
                ) = process_data(raw_data, target_frequency)
 
                excel_file = create_excel_file(clean_prices, log_returns)
 
            st.success("Elaborazione completata correttamente.")
 
            # =================================================
            # REPORT DIAGNOSTICO
            # =================================================
 
            st.markdown('<div id="metriche"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-label">Diagnostica degli asset</div>', unsafe_allow_html=True)
 
            diagnostic_display = diagnostic_report.copy()
 
            diagnostic_display["First valid date"] = (
                diagnostic_display["First valid date"].dt.strftime("%d/%m/%Y")
            )
            diagnostic_display["Last valid date"] = (
                diagnostic_display["Last valid date"].dt.strftime("%d/%m/%Y")
            )
 
            st.dataframe(diagnostic_display, use_container_width=True)
 
            # =================================================
            # METRICHE
            # =================================================
 
            st.markdown('<div class="sec-label">Risultati</div>', unsafe_allow_html=True)
 
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Asset", report["assets"])
            col2.metric("Righe iniziali", report["initial_rows"])
            col3.metric("Righe complete", report["clean_price_rows"])
            col4.metric("Log return", report["log_return_rows"])
 
            if report["frequency_warning"]:
                st.warning(report["frequency_warning"])
 
            st.write(
                f"""
                **Frequenza della griglia:** {report["frequency"]}
                (asset più rado: un'osservazione ogni ~{report["worst_native_gap"]:.0f} giorni)
                **Date non valide eliminate:** {report["invalid_dates"]}  
                **Date duplicate eliminate:** {report["duplicate_dates"]}  
                **Celle riempite con l'ultimo prezzo disponibile:** {report["filled_cells"]}  
                **Prima data comune disponibile:** {report["first_date"].strftime("%d/%m/%Y")}  
                **Ultima data comune disponibile:** {report["last_date"].strftime("%d/%m/%Y")}
                """
            )
 
            # =================================================
            # TABELLE
            # =================================================
 
            st.markdown('<div id="tabelle"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-label">Anteprima dei dati</div>', unsafe_allow_html=True)
 
            tab1, tab2 = st.tabs(["Clean Prices", "Log Returns"])
 
            with tab1:
                st.dataframe(clean_prices, use_container_width=True)
 
            with tab2:
                # Copia solo per la visualizzazione su Streamlit.
                log_returns_display = log_returns.copy()
 
                for column in log_returns_display.columns:
                    if column != "Date":
                        log_returns_display[column] = (
                            log_returns_display[column] * 100
                        )
 
                st.caption("I valori mostrati qui sono espressi in percentuale.")
 
                st.dataframe(
                    log_returns_display.style.format(
                        {
                            column: "{:.6f}%"
                            for column in log_returns_display.columns
                            if column != "Date"
                        }
                    ),
                    use_container_width=True,
                )
 
            # =================================================
            # DOWNLOAD
            # =================================================
 
            st.markdown('<div id="download"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-label">Download</div>', unsafe_allow_html=True)
 
            output_name = (
                uploaded_file.name.rsplit(".", 1)[0]
                + "_cleaned_prices_and_log_returns.xlsx"
            )
 
            st.download_button(
                label="Scarica il file Excel",
                data=excel_file,
                file_name=output_name,
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                type="primary",
                use_container_width=True,
            )
 
    except Exception as error:
        st.error(f"Errore durante l'elaborazione: {error}")
 
st.markdown(
    '<div class="sec-note">🔒 I tuoi dati sono al sicuro. Non memorizziamo nessun file.</div>',
    unsafe_allow_html=True,
)

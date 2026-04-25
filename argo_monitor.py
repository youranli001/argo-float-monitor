"""
argo_monitor.py — Argo Float Monitoring Dashboard
==================================================
Streamlit dashboard for float health and data delivery monitoring.
Built as a demonstration of PMEL-style fleet monitoring capability.

Usage:
    pip install streamlit xarray netcdf4 plotly scipy pandas numpy
    streamlit run argo_monitor.py

Author: Youran Li  |  youranli001 @ github
"""

import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

warnings.filterwarnings("ignore")


# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Argo Float Monitor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 1.3rem; font-weight: 600; }
    .stTabs [data-baseweb="tab"]  { font-size: 0.9rem; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
# ── GDAC download helpers (FTP: ftp.ifremer.fr) ───────────────────────────────
from ftplib import FTP

FTP_HOST   = "ftp.ifremer.fr"
FTP_BASE   = "/ifremer/argo/dac"
DAC_ORDER  = ["aoml", "pmel", "coriolis", "meds", "nmdis",
               "incois", "kordi", "bodc", "csio", "kma", "jma"]
FLOAT_FILES = ["_prof.nc", "_Sprof.nc", "_meta.nc", "_tech.nc",
               "_Dtraj.nc", "_Rtraj.nc"]
CACHE_DIR  = os.path.join(os.path.expanduser("~"), ".argo_cache")

@st.cache_data(show_spinner=False)
def find_dac_ftp(wmo: str) -> str | None:
    """Connect to IFREMER FTP and try each DAC to locate the float."""
    try:
        ftp = FTP(FTP_HOST, timeout=15)
        ftp.login()
        for dac in DAC_ORDER:
            try:
                ftp.cwd(f"{FTP_BASE}/{dac}/{wmo}")
                ftp.quit()
                return dac
            except Exception:
                continue
        ftp.quit()
    except Exception:
        pass
    return None

def download_float_files(wmo: str, dac: str, dest: str,
                         progress_bar) -> list[str]:
    """Download all standard files via FTP to dest/."""
    os.makedirs(dest, exist_ok=True)
    saved = []
    n = len(FLOAT_FILES)
    try:
        ftp = FTP(FTP_HOST, timeout=60)
        ftp.login()
        ftp.cwd(f"{FTP_BASE}/{dac}/{wmo}")
        remote_files = ftp.nlst()
        for i, suffix in enumerate(FLOAT_FILES):
            fname = wmo + suffix
            local = os.path.join(dest, fname)
            progress_bar.progress((i + 1) / n, text=f"  {fname}")
            if os.path.exists(local):
                saved.append(fname + " (cached)")
                continue
            if fname not in remote_files:
                saved.append(fname + " (not found)")
                continue
            try:
                with open(local, "wb") as f:
                    ftp.retrbinary(f"RETR {fname}", f.write)
                saved.append(fname)
            except Exception as e:
                saved.append(f"{fname} (error: {e})")
        ftp.quit()
    except Exception as e:
        progress_bar.empty()
        st.error(f"FTP error: {e}")
    return saved

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌊 Argo Float Monitor")
    st.divider()

    mode = st.radio("Data source", ["Local files", "Download from GDAC"],
                    index=0, horizontal=False)

    wmo = st.text_input("WMO number", value="5906551")

    if mode == "Local files":
        data_dir = st.text_input(
            "Data directory",
            value=r"C:\Users\Owner\OneDrive\Job_applications\Argo\5906551",
            help="Folder containing the float NetCDF files",
        )
        fetch_clicked = False
    else:
        data_dir = os.path.join(CACHE_DIR, wmo.strip())
        fetch_clicked = st.button("⬇ Download / refresh from GDAC",
                                  use_container_width=True)
        if os.path.isdir(data_dir):
            cached = [f for f in os.listdir(data_dir) if f.endswith(".nc")]
            st.caption(f"{len(cached)} files cached locally")

    st.divider()
    st.caption("Data: GDAC / IFREMER")
    st.caption("Format: Argo v3.1 NetCDF")

WMO = wmo.strip()
DIR = data_dir.rstrip(r"\/") + os.sep

# ── GDAC fetch (only runs when button clicked) ─────────────────────────────────
if mode == "Download from GDAC" and fetch_clicked:
    with st.spinner(f"Looking up float {WMO} on GDAC…"):
        dac = find_dac_ftp(WMO)
    if dac is None:
        st.error(f"Float {WMO} not found on GDAC. Check the WMO number.")
        st.stop()
    st.info(f"Found float {WMO} at DAC: **{dac}**")
    prog = st.progress(0, text="Starting download…")
    saved = download_float_files(WMO, dac, data_dir, prog)
    prog.empty()
    st.success(f"Downloaded {len(saved)} files → `{data_dir}`")
    st.cache_data.clear()  # force reload with new files


# ── Constants & helpers ────────────────────────────────────────────────────────
FILL   = 99999.0
JREF   = pd.Timestamp("1950-01-01")
C_BLUE = "#0077B6"
C_RED  = "#e63946"
C_GRN  = "#2dc653"
C_ORG  = "#f77f00"
C_PUR  = "#6a0dad"


def decode_bytes(arr: np.ndarray) -> np.ndarray:
    """Decode 1-D or 2-D bytes/str array → 1-D numpy str array."""
    if arr.ndim == 2:
        return np.array(
            ["".join(c.decode() if isinstance(c, bytes) else c for c in row).strip()
             for row in arr]
        )
    return np.array(
        [x.decode().strip() if isinstance(x, bytes) else str(x).strip()
         for x in arr]
    )


def mask_fill(arr) -> np.ndarray:
    """Replace fill values (≥ 99999) with NaN."""
    a = np.array(arr, dtype=float)
    a[a >= FILL] = np.nan
    return a


def juld_to_dates(arr) -> list:
    """Julian days (days since 1950-01-01) → list of pd.Timestamp / pd.NaT."""
    arr = np.array(arr, dtype=float)
    out = []
    for j in arr:
        if np.isnan(j) or j >= FILL:
            out.append(pd.NaT)
        else:
            out.append(JREF + pd.Timedelta(days=float(j)))
    return out


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_datasets(data_dir: str, wmo: str) -> dict:
    D = data_dir.rstrip(r"\/") + os.sep
    suffixes = {
        "prof":  "_prof.nc",
        "sprof": "_Sprof.nc",
        "tech":  "_tech.nc",
        "dtraj": "_Dtraj.nc",
        "rtraj": "_Rtraj.nc",
    }
    ds = {}
    for key, suffix in suffixes.items():
        path = D + wmo + suffix
        ds[key] = xr.open_dataset(path, decode_times=False) if os.path.exists(path) else None
    return ds


@st.cache_data(show_spinner=False)
def parse_tech(data_dir: str, wmo: str) -> pd.DataFrame | None:
    """Parse tech.nc long format → tidy DataFrame with columns [cycle, param, value]."""
    path = data_dir.rstrip(r"\/") + os.sep + wmo + "_tech.nc"
    if not os.path.exists(path):
        return None
    tech = xr.open_dataset(path, decode_times=False)
    names  = decode_bytes(tech["TECHNICAL_PARAMETER_NAME"].values)
    values = decode_bytes(tech["TECHNICAL_PARAMETER_VALUE"].values)
    cycles = tech["CYCLE_NUMBER"].values.astype(int)

    # CTD hex flags need separate handling — keep raw strings in a parallel column
    rows = []
    for name, val, cyc in zip(names, values, cycles):
        try:
            fval = float(val)
        except (ValueError, TypeError):
            fval = np.nan
        rows.append({"cycle": cyc, "param": name, "value": fval, "raw": val})
    return pd.DataFrame(rows)


def get_param(df: pd.DataFrame, param: str):
    """Extract one tech parameter → (cycles_sorted, values_sorted)."""
    sub = df[df["param"] == param].dropna(subset=["value"]).sort_values("cycle")
    return sub["cycle"].values, sub["value"].values


# ── Load all data ──────────────────────────────────────────────────────────────
with st.spinner("Loading float files…"):
    ds      = load_datasets(data_dir, wmo)
    df_tech = parse_tech(data_dir, wmo)

if all(v is None for v in ds.values()):
    st.error(
        "❌ No NetCDF files found. "
        "Check that the data directory contains files like `5906551_prof.nc`."
    )
    st.stop()


# ── Page header + top-level metrics ───────────────────────────────────────────
st.title(f"Float {WMO} — Monitoring Dashboard")

if ds["prof"] is not None:
    prof   = ds["prof"]
    n_prof = prof.dims["N_PROF"]
    dm     = decode_bytes(prof["DATA_MODE"].values)
    dates  = juld_to_dates(prof["JULD"].values)
    dates_s = [str(d)[:10] if pd.notna(d) else "—" for d in dates]

    n_d = int(np.sum(dm == "D"))
    n_a = int(np.sum(dm == "A"))
    valid_d = [d for d in dates if pd.notna(d)]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total cycles",    n_prof)
    c2.metric("Delayed-mode (D)", n_d,    help="DMQC complete — highest quality data")
    c3.metric("Adjusted (A)",     n_a,    help="Awaiting expert review — too recent for DMQC")
    c4.metric("First cycle",      valid_d[0].strftime("%Y-%m-%d")  if valid_d else "—")
    c5.metric("Latest cycle",     valid_d[-1].strftime("%Y-%m-%d") if len(valid_d) > 1 else "—")

st.divider()


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_track, tab_health, tab_delivery, tab_profiles, tab_bgc = st.tabs([
    "📍  Float Track",
    "🔧  Float Health",
    "📡  Data Delivery",
    "🌡️  Profile Explorer",
    "🧪  BGC Time Series",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Float Track
# ═══════════════════════════════════════════════════════════════════════════════
with tab_track:
    if ds["prof"] is None:
        st.warning("prof.nc not found in the specified directory.")
    else:
        lat    = mask_fill(prof["LATITUDE"].values)
        lon    = mask_fill(prof["LONGITUDE"].values)
        cycles = prof["CYCLE_NUMBER"].values.astype(int)
        dm_arr = decode_bytes(prof["DATA_MODE"].values)
        dates  = juld_to_dates(prof["JULD"].values)
        dates_s = [str(d)[:10] if pd.notna(d) else "—" for d in dates]

        valid = ~np.isnan(lat) & ~np.isnan(lon)

        hover_text = [
            f"Cycle {c}<br>{d_s}<br>({la:.3f}°, {lo:.3f}°)  [{md}]"
            for c, d_s, la, lo, md in zip(
                cycles[valid], np.array(dates_s)[valid],
                lat[valid], lon[valid], dm_arr[valid]
            )
        ]

        fig_map = go.Figure()
        fig_map.add_trace(go.Scattergeo(               # track line
            lon=lon[valid], lat=lat[valid], mode="lines",
            line=dict(width=1, color="rgba(0,119,182,0.3)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig_map.add_trace(go.Scattergeo(               # profile locations
            lon=lon[valid], lat=lat[valid], mode="markers",
            marker=dict(
                size=5, color=cycles[valid],
                colorscale="Plasma", showscale=True,
                colorbar=dict(title="Cycle", len=0.7),
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="Profiles",
        ))

        v_idx = np.where(valid)[0]
        for idx, label, sym, clr in [
            (v_idx[0],  "Start", "triangle-up",   "blue"),
            (v_idx[-1], "End",   "triangle-down",  "red"),
        ]:
            fig_map.add_trace(go.Scattergeo(
                lon=[lon[idx]], lat=[lat[idx]],
                mode="markers+text",
                marker=dict(size=12, symbol=sym, color=clr),
                text=[f"{label} ({dates_s[idx]})"],
                textposition="top right",
                name=label,
            ))

        lat_c   = float(np.nanmean(lat[valid]))
        lon_c   = float(np.nanmean(lon[valid]))
        lat_pad = max(float(np.nanstd(lat[valid])) * 3.5, 4)
        lon_pad = max(float(np.nanstd(lon[valid])) * 3.5, 6)

        fig_map.update_layout(
            geo=dict(
                projection_type="natural earth",
                showland=True,  landcolor="#f5f0eb",
                showocean=True, oceancolor="#cce5f5",
                showcoastlines=True, coastlinecolor="#999",
                lataxis_range=[lat_c - lat_pad, lat_c + lat_pad],
                lonaxis_range=[lon_c - lon_pad, lon_c + lon_pad],
            ),
            title=f"Float {WMO} — {valid.sum()} position fixes",
            height=560,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True)

        cc = st.columns(4)
        cc[0].metric("Lat range",   f"{np.nanmin(lat[valid]):.2f}° – {np.nanmax(lat[valid]):.2f}°")
        cc[1].metric("Lon range",   f"{np.nanmin(lon[valid]):.2f}° – {np.nanmax(lon[valid]):.2f}°")
        cc[2].metric("GPS fixes",   int(valid.sum()))
        cc[3].metric("Region",      "Southern Ocean")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Float Health (tech file)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_health:
    if df_tech is None:
        st.warning("tech.nc not found — float health analysis unavailable.")
    else:
        # ── Extract key parameters ─────────────────────────────────────────────
        cyc_pump,  pump_time = get_param(df_tech, "TIME_BuoyancyPumpOn_seconds")
        cyc_volt,  voltage   = get_param(df_tech, "VOLTAGE_BatteryPumpOn_volts")
        cyc_bat,   current   = get_param(df_tech, "CURRENT_BatteryPumpOn_mA")
        cyc_vac,   vacuum    = get_param(df_tech, "PRESSURE_InternalVacuum_inHg")
        cyc_pres,  pres_off  = get_param(df_tech, "PRES_SurfaceOffsetNotTruncated_dbar")
        cyc_repos, repos     = get_param(df_tech, "NUMBER_RepositionsDuringPark_COUNT")
        cyc_psurf, p_surf    = get_param(df_tech, "POSITION_PistonSurface_COUNT")
        cyc_ppark, p_park    = get_param(df_tech, "POSITION_PistonPark_COUNT")

        # CTD status flag — stored as hex string, needs separate extraction
        ctd_rows = df_tech[df_tech["param"] == "FLAG_CTDStatus_hex"].copy()
        ctd_rows["int_val"] = ctd_rows["raw"].apply(
            lambda v: int(v, 16) if isinstance(v, str) and v.strip() else 0
        )
        cyc_ctd  = ctd_rows["cycle"].values
        ctd_int  = ctd_rows["int_val"].values

        # ── Pump time linear trend ─────────────────────────────────────────────
        pump_slope = np.nan
        pump_trend = None
        if len(cyc_pump) > 5:
            slope, intercept, r_val, p_val, _ = stats.linregress(cyc_pump, pump_time)
            pump_slope  = slope
            pump_trend  = slope * cyc_pump + intercept

        # ── Summary row ────────────────────────────────────────────────────────
        st.subheader("Health Summary")

        n_ctd_bad    = int(np.sum(ctd_int > 0))
        n_high_repos = int(np.sum(repos > 5)) if len(repos) else 0
        mean_pres_off = float(np.nanmean(pres_off)) if len(pres_off) else np.nan

        cc = st.columns(4)
        cc[0].metric(
            "Pump time trend",
            f"{pump_slope:+.2f} s / cycle" if not np.isnan(pump_slope) else "—",
            help="Gradual increase = buoyancy system working harder over time",
        )
        cc[1].metric(
            "Mean pressure offset",
            f"{mean_pres_off:+.2f} dbar"  if not np.isnan(mean_pres_off) else "—",
            help="Should stay near 0; >±20 dbar triggers investigation",
        )
        cc[2].metric(
            "CTD error cycles",
            n_ctd_bad,
            delta="anomaly" if n_ctd_bad > 0 else None,
            delta_color="inverse",
            help="Cycles where CTD status hex flag ≠ 0x00",
        )
        cc[3].metric(
            "High-reposition cycles",
            n_high_repos,
            help="Cycles with >5 repositions — subsurface velocity estimate unreliable",
        )

        # Alert banners
        # Plain-text status summary (no colored boxes)
        status_lines = []
        if not np.isnan(pump_slope):
            icon = "⚠️" if pump_slope > 1.0 else "✓"
            status_lines.append(f"- {icon} Pump time trend: **{pump_slope:+.2f} s/cycle**"
                + (" — early buoyancy degradation signal" if pump_slope > 1.0 else " — normal"))
        if not np.isnan(mean_pres_off):
            icon = "⚠️" if abs(mean_pres_off) > 5 else "✓"
            status_lines.append(f"- {icon} Pressure offset: **{mean_pres_off:+.1f} dbar**"
                + (" — review against ±20 dbar threshold" if abs(mean_pres_off) > 5 else " — sensor healthy"))
        if n_ctd_bad > 0:
            status_lines.append(f"- ⚠️ CTD status flag anomalies on **{n_ctd_bad} cycles** — cross-check profile QC grades")
        if n_high_repos > 0:
            status_lines.append(f"- ⚠️ **{n_high_repos} cycles** with >5 repositions — subsurface velocity estimates unreliable")
        if status_lines:
            st.markdown("\n".join(status_lines))

        st.divider()

        # ── 6-panel engineering chart ──────────────────────────────────────────
        st.subheader(f"Engineering Telemetry — {len(df_tech['cycle'].unique())} cycles")

        fig_h = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Buoyancy Pump On-Time (s)",
                "Battery Voltage at Pump On (V)",
                "Battery Current at Pump On (mA)",
                "Internal Vacuum (inHg)",
                "Pressure Offset at Surface (dbar)",
                "Reposition Count per Cycle",
            ],
            vertical_spacing=0.13,
            horizontal_spacing=0.08,
        )

        # 1 — Pump time + trend
        if len(cyc_pump):
            fig_h.add_trace(go.Scatter(
                x=cyc_pump, y=pump_time, mode="markers+lines",
                marker=dict(size=4, color=C_BLUE), line=dict(width=1, color=C_BLUE),
                name="Pump time", showlegend=False,
            ), row=1, col=1)
        if pump_trend is not None:
            fig_h.add_trace(go.Scatter(
                x=cyc_pump, y=pump_trend, mode="lines",
                line=dict(width=2.5, color=C_RED, dash="dash"),
                name=f"Trend  {pump_slope:+.2f} s/cycle",
                showlegend=True,
            ), row=1, col=1)

        # 2 — Battery voltage
        if len(cyc_volt):
            fig_h.add_trace(go.Scatter(
                x=cyc_volt, y=voltage, mode="markers+lines",
                marker=dict(size=4, color=C_GRN), line=dict(width=1, color=C_GRN),
                showlegend=False,
            ), row=1, col=2)

        # 3 — Battery current
        if len(cyc_bat):
            fig_h.add_trace(go.Scatter(
                x=cyc_bat, y=current, mode="markers+lines",
                marker=dict(size=4, color=C_ORG), line=dict(width=1, color=C_ORG),
                showlegend=False,
            ), row=2, col=1)

        # 4 — Internal vacuum
        if len(cyc_vac):
            fig_h.add_trace(go.Scatter(
                x=cyc_vac, y=vacuum, mode="markers+lines",
                marker=dict(size=4, color=C_PUR), line=dict(width=1, color=C_PUR),
                showlegend=False,
            ), row=2, col=2)

        # 5 — Pressure offset + threshold lines
        if len(cyc_pres):
            fig_h.add_trace(go.Scatter(
                x=cyc_pres, y=pres_off, mode="markers+lines",
                marker=dict(size=4, color=C_BLUE), line=dict(width=1, color=C_BLUE),
                showlegend=False,
            ), row=3, col=1)
            x0, x1 = int(cyc_pres.min()), int(cyc_pres.max())
            for thresh in [20, -20]:
                fig_h.add_shape(
                    type="line", x0=x0, x1=x1, y0=thresh, y1=thresh,
                    line=dict(color=C_RED, dash="dot", width=1.5), row=3, col=1,
                )
            fig_h.add_annotation(
                x=x1, y=21, text="±20 dbar limit", showarrow=False,
                font=dict(color=C_RED, size=10), row=3, col=1,
            )

        # 6 — Reposition count (bar, red if high)
        if len(cyc_repos):
            bar_colors = [C_RED if r > 5 else C_BLUE for r in repos]
            fig_h.add_trace(go.Bar(
                x=cyc_repos, y=repos, marker_color=bar_colors,
                showlegend=False,
            ), row=3, col=2)
            fig_h.add_shape(
                type="line",
                x0=int(cyc_repos.min()), x1=int(cyc_repos.max()),
                y0=5, y1=5,
                line=dict(color=C_RED, dash="dot", width=1.5), row=3, col=2,
            )
            fig_h.add_annotation(
                x=int(cyc_repos.max()), y=6, text="vel. QC threshold",
                showarrow=False, font=dict(color=C_RED, size=10), row=3, col=2,
            )

        fig_h.update_xaxes(title_text="Cycle number")
        fig_h.update_layout(height=820, showlegend=True, legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig_h, use_container_width=True)

        # CTD flag detail table
        if n_ctd_bad > 0:
            st.subheader("CTD Status Flag Anomalies")
            bad = ctd_rows[ctd_rows["int_val"] > 0]
            st.dataframe(
                bad[["cycle", "raw", "int_val"]].rename(columns={
                    "cycle": "Cycle", "raw": "Hex flag", "int_val": "Integer value"
                }),
                use_container_width=True,
            )
            st.caption(
                "Non-zero CTD status = error condition from APEX firmware. "
                "Exact bit meaning requires manufacturer documentation (not publicly released). "
                "Cross-check these cycles against profile-level QC grades in prof.nc."
            )

            # Piston gap = surface position - park position (narrowing gap = bad)
        if len(cyc_psurf) and len(cyc_ppark):
            st.subheader("Piston Gap (Surface − Park position, stepper counts)")
            common = np.intersect1d(cyc_psurf, cyc_ppark)
            if len(common):
                s_vals = np.array([p_surf[cyc_psurf == c][0] for c in common])
                p_vals = np.array([p_park[cyc_ppark == c][0] for c in common])
                gap    = s_vals - p_vals
                fig_gap = go.Figure(go.Scatter(
                    x=common, y=gap, mode="markers+lines",
                    marker=dict(size=4, color=C_BLUE), line=dict(width=1, color=C_BLUE),
                ))
                fig_gap.update_layout(
                    xaxis_title="Cycle", yaxis_title="Stepper count gap",
                    title="Piston gap — narrowing trend = piston cannot fully extend → float may fail to surface",
                    height=280,
                )
                st.plotly_chart(fig_gap, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Data Delivery
# ═══════════════════════════════════════════════════════════════════════════════
with tab_delivery:
    traj = ds["dtraj"] if ds["dtraj"] is not None else ds["rtraj"]

    if traj is None:
        st.warning("No trajectory file found (Dtraj.nc / Rtraj.nc).")
    else:
        TARGET_H = 12.0   # Argo 12-hour delivery target

        # ── Timing from N_CYCLE dimension ──────────────────────────────────────
        # JULD_ASCENT_END = when float finished ascending (reached surface)
        # JULD_TRANSMISSION_START = when Iridium transmission began
        # Difference (× 24) = surface overhead time in hours
        asc_end  = mask_fill(traj["JULD_ASCENT_END"].values)
        tx_start = mask_fill(traj["JULD_TRANSMISSION_START"].values)

        # Build cycle index for N_CYCLE dimension
        # Some files have CYCLE_NUMBER_INDEX; fall back to sequential index
        n_cyc = traj.dims["N_CYCLE"]
        try:
            cyc_idx = traj["CYCLE_NUMBER_INDEX"].values.astype(int)
        except KeyError:
            cyc_idx = np.arange(1, n_cyc + 1)

        valid_t = ~np.isnan(asc_end) & ~np.isnan(tx_start)
        delay_h = (tx_start[valid_t] - asc_end[valid_t]) * 24.0
        cyc_t   = cyc_idx[valid_t]

        # Filter implausible values
        plaus   = (delay_h >= 0) & (delay_h < 120)
        delay_h = delay_h[plaus]
        cyc_t   = cyc_t[plaus]

        if len(delay_h) == 0:
            st.info(
                "Transmission timing variables (JULD_ASCENT_END, JULD_TRANSMISSION_START) "
                "are not populated in this file. This is common for older Argos floats."
            )
        else:
            n_on  = int(np.sum(delay_h <= TARGET_H))
            pct   = n_on / len(delay_h) * 100

            st.subheader("Surface-to-Transmit Delay per Cycle")
            st.caption(
                "Delay = time from ascent end (float at surface) to first Iridium transmission. "
                "Argo real-time target: data at GDAC within **12 hours** of surfacing."
            )

            cc = st.columns(4)
            cc[0].metric("Cycles with timing", len(delay_h))
            cc[1].metric("On-time (< 12 h)",   f"{pct:.0f}%",
                         delta_color="normal" if pct >= 90 else "inverse")
            cc[2].metric("Median delay",        f"{np.median(delay_h):.1f} h")
            cc[3].metric("Max delay",           f"{np.max(delay_h):.1f} h")

            bar_colors = [C_GRN if d <= TARGET_H else C_RED for d in delay_h]

            fig_del = go.Figure()
            fig_del.add_trace(go.Bar(
                x=cyc_t, y=delay_h, marker_color=bar_colors,
                hovertemplate="Cycle %{x}<br>Delay: %{y:.1f} h<extra></extra>",
                showlegend=False,
            ))
            fig_del.add_shape(
                type="line",
                x0=int(cyc_t.min()), x1=int(cyc_t.max()),
                y0=TARGET_H, y1=TARGET_H,
                line=dict(color=C_RED, dash="dash", width=2),
            )
            fig_del.add_annotation(
                x=int(cyc_t.max()), y=TARGET_H + 1.5,
                text="12-hour target", showarrow=False,
                font=dict(color=C_RED, size=11),
            )
            fig_del.update_layout(
                title=f"Float {WMO} — Transmission Delay",
                xaxis_title="Cycle number",
                yaxis_title="Delay (hours)",
                height=420,
            )
            st.plotly_chart(fig_del, use_container_width=True)

            # Delay distribution histogram
            fig_hist = px.histogram(
                x=delay_h, nbins=25,
                labels={"x": "Delay (hours)", "y": "Count"},
                title="Delay Distribution",
                color_discrete_sequence=[C_BLUE],
            )
            fig_hist.add_vline(
                x=TARGET_H, line_dash="dash", line_color=C_RED,
                annotation_text="12-h target", annotation_position="top right",
            )
            fig_hist.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Subsurface velocity QC — reposition count ──────────────────────────
        st.divider()
        st.subheader("Subsurface Velocity Quality — Reposition Count")
        st.markdown("""
Argo derives subsurface parking-depth velocity **u, v** from passive displacement:
the float drifts at 1000 m for ~9 days, and velocity is estimated from the distance
between consecutive surface GPS fixes divided by elapsed time.

If the float **repositioned** during parking (pump fired to return to target depth),
the drift was not passive — **the velocity estimate is contaminated for that cycle**.
There is no dedicated velocity QC flag in the Argo format; assessing velocity
quality requires cross-referencing the tech file with the trajectory file.
        """)

        if df_tech is not None:
            cyc_rep, rep_vals = get_param(df_tech, "NUMBER_RepositionsDuringPark_COUNT")
            bad_cycles = cyc_rep[rep_vals > 0]
            n_bad      = len(bad_cycles)

            cc2 = st.columns(3)
            cc2[0].metric("Cycles with ≥1 reposition", n_bad)
            cc2[1].metric("Velocity contamination rate", f"{n_bad/max(len(cyc_rep),1)*100:.0f}%")
            cc2[2].metric("Max repositions (single cycle)", int(np.nanmax(rep_vals)) if len(rep_vals) else 0)

            if n_bad:
                st.markdown(
                    f"**Cycles {sorted(bad_cycles.tolist()[:20])}**"
                    + (" …" if n_bad > 20 else "")
                    + " — velocity estimates should be flagged for these cycles."
                )

            fig_rep = go.Figure(go.Bar(
                x=cyc_rep, y=rep_vals,
                marker_color=[C_RED if r > 0 else C_BLUE for r in rep_vals],
                hovertemplate="Cycle %{x}<br>Repositions: %{y}<extra></extra>",
            ))
            fig_rep.add_shape(
                type="line",
                x0=int(cyc_rep.min()), x1=int(cyc_rep.max()),
                y0=1, y1=1,
                line=dict(color=C_RED, dash="dot", width=1.5),
            )
            fig_rep.update_layout(
                title="Reposition Count per Cycle  (red = velocity estimate unreliable)",
                xaxis_title="Cycle number",
                yaxis_title="Reposition count",
                height=300,
            )
            st.plotly_chart(fig_rep, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Profile Explorer
# ═══════════════════════════════════════════════════════════════════════════════
with tab_profiles:
    # Prefer Sprof (core + BGC), fall back to prof (core only)
    pds_key  = "sprof" if ds["sprof"] is not None else "prof"
    prof_ds  = ds[pds_key]

    if prof_ds is None:
        st.warning("No profile file found.")
    else:
        n_prof   = prof_ds.dims["N_PROF"]
        cycles   = prof_ds["CYCLE_NUMBER"].values.astype(int)
        # Sprof.nc has PARAMETER_DATA_MODE (N_PROF, N_PARAM) instead of DATA_MODE (N_PROF,)
        # Use the first parameter's mode (PRES) as the representative profile mode
        if "DATA_MODE" in prof_ds:
            dm = decode_bytes(prof_ds["DATA_MODE"].values)
        elif "PARAMETER_DATA_MODE" in prof_ds:
            dm = decode_bytes(prof_ds["PARAMETER_DATA_MODE"].values[:, 0])
        else:
            dm = np.array(["?" ] * n_prof)
        dates    = juld_to_dates(prof_ds["JULD"].values)
        dates_s  = [str(d)[:10] if pd.notna(d) else "—" for d in dates]

        is_sprof = pds_key == "sprof"
        has_bgc  = is_sprof and any(
            p in prof_ds for p in ["DOXY", "DOXY_ADJUSTED", "CHLA", "CHLA_ADJUSTED"]
        )

        st.caption(f"File: `{WMO}_{pds_key.replace('sprof','Sprof').replace('prof','prof')}.nc`"
                   f"  ·  {'Core + BGC (Sprof)' if is_sprof else 'Core only (prof)'}")

        # Controls
        cc_ctrl = st.columns([2, 5])
        with cc_ctrl[0]:
            sel_cycle = st.selectbox(
                "Select cycle",
                options=cycles.tolist(),
                format_func=lambda c: (
                    f"Cycle {c:3d}  [{dm[np.where(cycles == c)[0][0]]}]  "
                    f"{dates_s[np.where(cycles == c)[0][0]]}"
                ),
            )
        i_sel    = int(np.where(cycles == sel_cycle)[0][0])
        mode_sel = dm[i_sel]
        mode_labels = {"D": "DMQC complete", "A": "Adjusted (awaiting expert review)", "R": "Real-time only"}
        with cc_ctrl[1]:
            st.markdown(
                f"**Cycle {sel_cycle}**  ·  DATA_MODE: `{mode_sel}`  "
                f"({mode_labels.get(mode_sel, mode_sel)})  ·  {dates_s[i_sel]}"
            )
            if mode_sel == "A":
                st.caption("ℹ️ A-mode: use PARAM_ADJUSTED fields — automated correction applied but not expert-reviewed.")
            elif mode_sel == "D":
                st.caption("✅ D-mode: PARAM_ADJUSTED fields are DMQC-corrected — use these for science.")

        # Helper: extract one variable for selected cycle
        def get_var(name):
            if name in prof_ds:
                return mask_fill(prof_ds[name].values[i_sel])
            return None

        pres_raw = get_var("PRES")
        _pres_adj = get_var("PRES_ADJUSTED")
        pres_adj = _pres_adj if _pres_adj is not None else pres_raw
        temp_raw = get_var("TEMP")
        temp_adj = get_var("TEMP_ADJUSTED")
        psal_raw = get_var("PSAL")
        psal_adj = get_var("PSAL_ADJUSTED")

        def valid_mask(x, p):
            if x is None or p is None:
                return None
            return ~np.isnan(x) & ~np.isnan(p)

        # ── Row 1a: Temperature and Salinity depth profiles ────────────────
        fig_ts = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Temperature (°C)", "Salinity (PSU)"],
            shared_yaxes=True,
            horizontal_spacing=0.08,
        )

        def add_profile(fig, col, x_raw, x_adj, p_raw, p_adj, c_raw, c_adj, label):
            m_r = valid_mask(x_raw, p_raw)
            if m_r is not None and m_r.any():
                fig.add_trace(go.Scatter(
                    x=x_raw[m_r], y=p_raw[m_r], mode="lines",
                    line=dict(color=c_raw, width=1, dash="dot"),
                    name=f"{label} raw",
                ), row=1, col=col)
            m_a = valid_mask(x_adj, p_adj)
            if m_a is not None and m_a.any():
                fig.add_trace(go.Scatter(
                    x=x_adj[m_a], y=p_adj[m_a], mode="lines+markers",
                    marker=dict(size=3, color=c_adj),
                    line=dict(color=c_adj, width=2),
                    name=f"{label} adjusted",
                ), row=1, col=col)

        add_profile(fig_ts, 1, temp_raw, temp_adj, pres_raw, pres_adj,
                    "#a8c8e8", C_BLUE, "Temp")
        add_profile(fig_ts, 2, psal_raw, psal_adj, pres_raw, pres_adj,
                    "#f4b8a0", C_RED, "Sal")

        fig_ts.update_yaxes(autorange="reversed", title_text="Pressure (dbar)", col=1)
        fig_ts.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
        fig_ts.update_xaxes(title_text="Salinity (PSU)", row=1, col=2)
        fig_ts.update_layout(
            height=400,
            title=f"Cycle {sel_cycle}  ·  {mode_sel}-mode  ·  {dates_s[i_sel]}",
            showlegend=True,
            legend=dict(orientation="h", y=-0.2, font=dict(size=11)),
            margin=dict(b=80),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # ── Row 1b: Classic T-S diagram (separate figure — independent axes) ─
        t_plot = temp_adj if temp_adj is not None else temp_raw
        s_plot = psal_adj if psal_adj is not None else psal_raw
        m_ts   = valid_mask(t_plot, s_plot)
        if m_ts is not None and m_ts.any():
            p_col  = pres_adj if pres_adj is not None else pres_raw
            fig_tsdiag = go.Figure()
            fig_tsdiag.add_trace(go.Scatter(
                x=s_plot[m_ts], y=t_plot[m_ts],
                mode="markers",
                marker=dict(size=4, color=p_col[m_ts],
                            colorscale="Blues_r", showscale=False),
                hovertemplate="S=%{x:.3f}  T=%{y:.2f}°C  P=%{marker.color:.0f} dbar<extra></extra>",
            ))
            fig_tsdiag.update_layout(
                title="T-S Diagram — water mass fingerprint",
                xaxis_title="Salinity (PSU)",
                yaxis_title="Temperature (°C)",
                height=380,
                margin=dict(t=50, b=50),
            )
            st.plotly_chart(fig_tsdiag, use_container_width=True)

        # ── Row 2: BGC parameters ──────────────────────────────────────────
        if has_bgc:
            bgc_spec = [
                ("DOXY_ADJUSTED",                "O₂ (µmol/kg)",  C_GRN),
                ("CHLA_ADJUSTED",                "Chl-a (mg/m³)", "#2a9d8f"),
                ("NITRATE_ADJUSTED",             "NO₃⁻ (µmol/kg)", C_ORG),
                ("PH_IN_SITU_TOTAL_ADJUSTED",    "pH",             C_RED),
                ("BBP700_ADJUSTED",              "BBP700 (m⁻¹)",  C_PUR),
            ]
            avail_bgc = [(v, u, c) for v, u, c in bgc_spec if v in prof_ds or
                         v.replace("_ADJUSTED","") in prof_ds]
            # resolve to actual variable name in file
            resolved = []
            for v, u, c in avail_bgc:
                actual = v if v in prof_ds else v.replace("_ADJUSTED","")
                resolved.append((actual, u, c))

            if resolved:
                fig_row2 = make_subplots(
                    rows=1, cols=len(resolved),
                    subplot_titles=[u for _, u, _ in resolved],
                    shared_yaxes=True,
                    horizontal_spacing=0.05,
                )
                for col_j, (var, unit, color) in enumerate(resolved, start=1):
                    vals = get_var(var)
                    m = valid_mask(vals, pres_adj)
                    if m is not None and m.any():
                        fig_row2.add_trace(go.Scatter(
                            x=vals[m], y=pres_adj[m],
                            mode="lines+markers",
                            marker=dict(size=3, color=color),
                            line=dict(color=color, width=1.5),
                            name=var, showlegend=False,
                        ), row=1, col=col_j)
                    fig_row2.update_xaxes(title_text=unit, row=1, col=col_j,
                                          title_font=dict(size=10))

                fig_row2.update_yaxes(autorange="reversed",
                                      title_text="Pressure (dbar)", col=1)
                fig_row2.update_layout(
                    height=400,
                    title=f"Cycle {sel_cycle}  ·  {mode_sel}-mode  — BGC",
                    showlegend=False,
                    margin=dict(b=60),
                )
                st.plotly_chart(fig_row2, use_container_width=True)


        # ── All profiles T/S overlay ───────────────────────────────────────────
        with st.expander("📈 All profiles — T/S overlay (all cycles)", expanded=False):
            st.caption(
                "All cycles overlaid, colored blue→red (early→late). "
                "Pattern shows water mass evolution over the float lifetime."
            )
            import plotly.colors as pc
            pds_all = ds["prof"] if ds["prof"] is not None else prof_ds
            n_all   = pds_all.dims["N_PROF"]
            cscale  = pc.sample_colorscale("RdYlBu_r", n_all)

            fig_all = make_subplots(
                rows=1, cols=3,
                subplot_titles=["Temperature (°C)", "Salinity (PSU)", "T-S Diagram"],
                shared_yaxes=False, horizontal_spacing=0.08,
            )
            for i in range(n_all):
                pres = pds_all["PRES_ADJUSTED"].values[i, :].astype(float)
                temp = pds_all["TEMP_ADJUSTED"].values[i, :].astype(float)
                psal = pds_all["PSAL_ADJUSTED"].values[i, :].astype(float)
                mt  = (temp < 9999) & (pres < 9999) & (temp > -9999)
                ms  = (psal < 9999) & (pres < 9999) & (psal > -9999)
                mts = mt & ms
                clr = cscale[i]
                cyc_lbl = int(pds_all["CYCLE_NUMBER"].values[i])
                ht_t  = f"Cycle {cyc_lbl}<br>T=%{{x:.2f}}°C P=%{{y:.0f}}dbar<extra></extra>"
                ht_s  = f"Cycle {cyc_lbl}<br>S=%{{x:.3f}} P=%{{y:.0f}}dbar<extra></extra>"
                ht_ts = f"Cycle {cyc_lbl}<br>S=%{{x:.3f}} T=%{{y:.2f}}°C<extra></extra>"
                if mt.any():
                    fig_all.add_trace(go.Scatter(
                        x=temp[mt], y=pres[mt], mode="lines",
                        line=dict(color=clr, width=0.6), opacity=0.35,
                        showlegend=False, hovertemplate=ht_t,
                    ), row=1, col=1)
                if ms.any():
                    fig_all.add_trace(go.Scatter(
                        x=psal[ms], y=pres[ms], mode="lines",
                        line=dict(color=clr, width=0.6), opacity=0.35,
                        showlegend=False, hovertemplate=ht_s,
                    ), row=1, col=2)
                if mts.any():
                    fig_all.add_trace(go.Scatter(
                        x=psal[mts], y=temp[mts], mode="lines",
                        line=dict(color=clr, width=0.6), opacity=0.35,
                        showlegend=False, hovertemplate=ht_ts,
                    ), row=1, col=3)
            fig_all.update_yaxes(autorange="reversed", title_text="Pressure (dbar)", col=1)
            fig_all.update_yaxes(autorange="reversed", col=2)
            fig_all.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
            fig_all.update_xaxes(title_text="Salinity (PSU)", row=1, col=2)
            fig_all.update_xaxes(title_text="Salinity (PSU)", row=1, col=3)
            fig_all.update_yaxes(title_text="Temperature (°C)", row=1, col=3)
            fig_all.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                marker=dict(color=cscale[0], size=10), name="Early (cycle 1)"))
            fig_all.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                marker=dict(color=cscale[-1], size=10), name=f"Late (cycle {n_all})"))
            fig_all.update_layout(height=520,
                title=f"Float {WMO} — {n_all} profiles (blue=early, red=late)",
                showlegend=True, legend=dict(orientation="h", y=-0.12))
            st.plotly_chart(fig_all, use_container_width=True)

        # Calibration records
        if "SCIENTIFIC_CALIB_EQUATION" in prof_ds:
            with st.expander("📋 SCIENTIFIC_CALIB records for this cycle"):
                try:
                    params_c = decode_bytes(prof_ds["PARAMETER"].values[i_sel, 0, :])
                    eq_c     = decode_bytes(prof_ds["SCIENTIFIC_CALIB_EQUATION"].values[i_sel, 0, :])
                    coeff_c  = decode_bytes(prof_ds["SCIENTIFIC_CALIB_COEFFICIENT"].values[i_sel, 0, :])
                    comm_c   = decode_bytes(prof_ds["SCIENTIFIC_CALIB_COMMENT"].values[i_sel, 0, :])
                    calib_df = pd.DataFrame({
                        "Parameter":   params_c,
                        "Equation":    eq_c,
                        "Coefficient": coeff_c,
                        "Comment":     comm_c,
                    })
                    calib_df = calib_df[
                        calib_df["Parameter"].str.len() > 0
                    ].reset_index(drop=True)
                    st.dataframe(calib_df, use_container_width=True)
                    st.caption(
                        "These records are written by the DMQC expert and document exactly "
                        "what adjustment was applied to each parameter."
                    )
                except Exception as e:
                    st.caption(f"Could not extract calibration records: {e}")

        # DATA_MODE timeline for all cycles
        with st.expander("📊 DATA_MODE timeline — all cycles"):
            mode_colors = {"D": C_BLUE, "A": C_ORG, "R": C_RED}
            color_list  = [mode_colors.get(m, "gray") for m in dm]

            fig_dm = go.Figure(go.Scatter(
                x=cycles, y=[1] * len(cycles),
                mode="markers",
                marker=dict(color=color_list, size=9, symbol="square"),
                hovertemplate="Cycle %{x}  DATA_MODE=%{text}<extra></extra>",
                text=dm,
            ))
            # Add legend traces manually
            for label, clr, md in [("D — delayed-mode", C_BLUE, "D"),
                                    ("A — adjusted",    C_ORG,  "A"),
                                    ("R — real-time",   C_RED,  "R")]:
                fig_dm.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(color=clr, size=9, symbol="square"),
                    name=label,
                ))
            fig_dm.update_layout(
                height=160, yaxis_visible=False,
                xaxis_title="Cycle number",
                title="DATA_MODE per cycle",
                showlegend=True,
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_dm, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — BGC Time Series (from trajectory file)
# Directly adapted from notebook Section 10 — BGC Parameters along trajectory
# ═══════════════════════════════════════════════════════════════════════════════
with tab_bgc:
    traj_bgc = ds["dtraj"] if ds["dtraj"] is not None else ds["rtraj"]

    if traj_bgc is None:
        st.warning("No trajectory file found (Dtraj.nc / Rtraj.nc).")
    else:
        st.subheader("BGC Parameters Along Trajectory")
        st.caption(
            "Intermediate measurements sampled throughout each cycle (N_MEASUREMENT dimension), "
            "including dense deep sampling at max depth (MC=290) used for sensor calibration. "
            "Raw vs adjusted shown where DMQC has been applied."
        )

        # ── Convert JULD → datetime  (exact pattern from notebook) ────────────
        juld_raw = traj_bgc["JULD"].values.astype(float)
        dates_traj = juld_to_dates(juld_raw)
        dates_arr  = np.array(dates_traj, dtype=object)

        # valid date mask
        valid_date = np.array([d is not pd.NaT and d is not None for d in dates_arr])

        # ── BGC parameter list: (raw_var, adj_var, unit, color) ──────────────
        bgc_traj_params = [
            ("DOXY",      "DOXY_ADJUSTED",      "µmol/kg", C_BLUE),
            ("CHLA",      "CHLA_ADJUSTED",       "mg/m³",   C_GRN),
            ("BBP700",    "BBP700_ADJUSTED",     "m⁻¹",     C_ORG),
            ("PPOX_DOXY", "PPOX_DOXY_ADJUSTED",  "mbar",    C_RED),
        ]

        # keep only params that exist
        bgc_traj_params = [
            (r, a, u, c) for r, a, u, c in bgc_traj_params if r in traj_bgc
        ]

        if not bgc_traj_params:
            st.info("No BGC parameters found in trajectory file.")
        else:
            # count valid records per param
            cc_bgc = st.columns(len(bgc_traj_params))
            for col_i, (raw_var, adj_var, unit, _) in enumerate(bgc_traj_params):
                raw = traj_bgc[raw_var].values.astype(float)
                n_valid = int(np.sum((raw < 99999) & valid_date))
                cc_bgc[col_i].metric(raw_var, f"{n_valid:,} records")

            # ── 4-panel time series (one per BGC parameter) ───────────────────
            fig_bgc = make_subplots(
                rows=len(bgc_traj_params), cols=1,
                subplot_titles=[f"{r}  ({u})" for r, _, u, _ in bgc_traj_params],
                shared_xaxes=True,
                vertical_spacing=0.06,
            )

            for row_i, (raw_var, adj_var, unit, color) in enumerate(bgc_traj_params, start=1):
                raw = traj_bgc[raw_var].values.astype(float)
                raw_mask = (raw < 99999) & valid_date

                if raw_mask.any():
                    fig_bgc.add_trace(go.Scatter(
                        x=dates_arr[raw_mask],
                        y=raw[raw_mask],
                        mode="markers",
                        marker=dict(size=2, color=color, opacity=0.5),
                        name=f"{raw_var} raw",
                        showlegend=(row_i == 1),
                        legendgroup="raw",
                    ), row=row_i, col=1)

                if adj_var in traj_bgc:
                    adj = traj_bgc[adj_var].values.astype(float)
                    adj_mask = (adj < 99999) & valid_date
                    if adj_mask.any():
                        fig_bgc.add_trace(go.Scatter(
                            x=dates_arr[adj_mask],
                            y=adj[adj_mask],
                            mode="markers",
                            marker=dict(size=2, color="black", opacity=0.8),
                            name=f"adjusted",
                            showlegend=(row_i == 1),
                            legendgroup="adj",
                        ), row=row_i, col=1)

                fig_bgc.update_yaxes(title_text=unit, row=row_i, col=1)

            fig_bgc.update_layout(
                height=200 * len(bgc_traj_params),
                title=f"Float {WMO} — BGC parameters along trajectory",
                showlegend=True,
                legend=dict(orientation="h", y=1.02),
            )
            fig_bgc.update_xaxes(title_text="Date", row=len(bgc_traj_params), col=1)
            st.plotly_chart(fig_bgc, use_container_width=True)

            st.caption(
                "Dense clusters at regular intervals = MC=290 deep sampling "
                "(float at max depth for BGC sensor calibration — not transmitted in profile files). "
                "Black points = adjusted values where DMQC has been applied."
            )

            # ── PPOX_DOXY in-air calibration note ────────────────────────────
            if "PPOX_DOXY" in traj_bgc:
                st.markdown("""
**PPOX_DOXY — in-air O₂ calibration**

Every time the float surfaces, it measures O₂ partial pressure in air before submerging.
Atmospheric O₂ is globally constant (~0.2095 atm × local barometric pressure),
giving a free calibration point every cycle. This is used to correct for
optode drift over the float lifetime — visible as the systematic adjustment between
raw and black (adjusted) points above.
                """)

        # ── BGC depth profiles for one selected cycle ────────────────────────
        st.divider()
        st.subheader("BGC Depth Profiles — Selected Cycle")
        st.caption("From Sprof.nc — BGC parameters vs pressure for one cycle.")

        sprof_bgc = ds["sprof"]
        if sprof_bgc is None:
            st.info("Sprof.nc not found.")
        else:
            cyc_bgc = sprof_bgc["CYCLE_NUMBER"].values.astype(int)
            sel_bgc = st.selectbox(
                "Select cycle", cyc_bgc.tolist(), key="bgc_cycle_sel",
                index=0,
            )
            i_bgc = int(np.where(cyc_bgc == sel_bgc)[0][0])

            bgc_profile_vars = [
                ("DOXY_ADJUSTED",       "O₂ (µmol/kg)",   C_BLUE),
                ("CHLA_ADJUSTED",       "Chl-a (mg/m³)",  C_GRN),
                ("NITRATE_ADJUSTED",    "NO₃⁻ (µmol/kg)", C_ORG),
                ("PH_IN_SITU_TOTAL_ADJUSTED",    "pH",     C_RED),
                ("BBP700_ADJUSTED",     "BBP700 (m⁻¹)",   C_PUR),
            ]
            avail = [(v, u, c) for v, u, c in bgc_profile_vars if v in sprof_bgc]

            if not avail:
                st.info("No BGC variables found in Sprof.nc.")
            else:
                pres_bgc = sprof_bgc["PRES_ADJUSTED"].values[i_bgc, :].astype(float)
                pres_bgc[pres_bgc >= 99999] = np.nan

                fig_bgc_prof = make_subplots(
                    rows=1, cols=len(avail),
                    subplot_titles=[u for _, u, _ in avail],
                    shared_yaxes=True,
                    horizontal_spacing=0.04,
                )
                for col_i, (var, unit, color) in enumerate(avail, start=1):
                    vals = sprof_bgc[var].values[i_bgc, :].astype(float)
                    vals[vals >= 99999] = np.nan
                    mask = ~np.isnan(vals) & ~np.isnan(pres_bgc)
                    if mask.any():
                        fig_bgc_prof.add_trace(go.Scatter(
                            x=vals[mask], y=pres_bgc[mask],
                            mode="lines+markers",
                            marker=dict(size=3, color=color),
                            line=dict(color=color, width=1.5),
                            name=var, showlegend=False,
                        ), row=1, col=col_i)
                    fig_bgc_prof.update_xaxes(title_text=unit, row=1, col=col_i)

                fig_bgc_prof.update_yaxes(
                    autorange="reversed",
                    title_text="Pressure (dbar)",
                    col=1,
                )
                fig_bgc_prof.update_layout(
                    height=500,
                    title=f"Float {WMO} — Cycle {sel_bgc} BGC profiles (ADJUSTED)",
                )
                st.plotly_chart(fig_bgc_prof, use_container_width=True)

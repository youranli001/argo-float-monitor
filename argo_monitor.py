"""
argo_monitor.py — Argo Float Monitoring Dashboard
==================================================
Streamlit dashboard for float health and data delivery monitoring.
Built as a demonstration of PMEL-style fleet monitoring capability.

Usage:
    pip install streamlit xarray netcdf4 plotly scipy pandas numpy
    streamlit run argo_monitor.py

Author: Youran Li  |  youranli001 @ github

Tab structure (Batches A + B complete):
    1. Main Information       — map + 5 text cards
    2. Technical Details      — sensors + 6-panel engineering telemetry
    3. Profiles & Sections    — section plots + overlay grid + single-cycle explorer
    4. QC & Processing        — per-parameter QC section + QC-colored overlay
    5. Data Delivery          — variables table + delay scatter + DM eligibility + velocity QC
    6. BGC Time Series        — trajectory time series + per-cycle BGC profiles
"""

import os
import warnings
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
from plotly.subplots import make_subplots
from scipy import stats
from scipy.interpolate import interp1d

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


# ── Detect Streamlit Cloud environment ────────────────────────────────────────
IS_CLOUD = os.getenv('USER') == 'appuser'

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌊 Argo Float Monitor")
    st.divider()

    if IS_CLOUD:
        mode = "Download from GDAC"
    else:
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


# ═══════════════════════════════════════════════════════════════════════════════
# ── BATCH A: NEW HELPERS ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def d(x):
    """Decode single bytes/str/scalar to clean string. Robust to 0-d arrays."""
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()
    if isinstance(x, bytes):
        return x.decode().strip()
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ''
    return str(x).strip()


def parse_argo_date(s):
    """'20221220234800' (YYYYMMDDHHMMSS) → datetime, '' → None."""
    s = d(s)
    if not s or s in ('n/a', 'nan'):
        return None
    try:
        return datetime.strptime(s[:14], '%Y%m%d%H%M%S')
    except ValueError:
        return None


def juld_to_dt(juld_val, ref=datetime(1950, 1, 1)):
    """Single JULD float → datetime. Returns None on FillValue."""
    if np.isnan(juld_val) or juld_val > 999990:
        return None
    return ref + timedelta(days=float(juld_val))


def decode_qc(qc_arr):
    """QC bytes → int array. Empty/missing entries become 9."""
    out = np.full(qc_arr.shape, 9, dtype=np.int8)
    for idx in np.ndindex(qc_arr.shape):
        v = qc_arr[idx]
        s = v.decode().strip() if isinstance(v, bytes) else str(v).strip()
        if s.isdigit():
            out[idx] = int(s)
    return out


def get_best(ds, param):
    """Return PARAM_ADJUSTED, falling back to PARAM where adjusted is missing."""
    adj_name = f'{param}_ADJUSTED'
    if adj_name in ds:
        adj = mask_fill(ds[adj_name].values)
        if param in ds:
            raw = mask_fill(ds[param].values)
            adj = np.where(np.isnan(adj), raw, adj)
        return adj
    if param in ds:
        return mask_fill(ds[param].values)
    return None


def get_qc(ds, param):
    """QC array (N_PROF, N_LEVELS) as int. Prefers ADJUSTED_QC."""
    for name in (f'{param}_ADJUSTED_QC', f'{param}_QC'):
        if name in ds:
            return decode_qc(ds[name].values)
    return None


def get_data_mode_per_param(ds, param):
    """Per-parameter data-mode array (length N_PROF)."""
    if 'PARAMETER_DATA_MODE' in ds and 'STATION_PARAMETERS' in ds:
        # Use first profile's STATION_PARAMETERS to find the param index
        sp_row = ds['STATION_PARAMETERS'].values[0]
        sp_str = []
        for s in sp_row:
            if isinstance(s, bytes):
                sp_str.append(s.decode().strip())
            elif isinstance(s, np.ndarray):
                try:
                    sp_str.append(b''.join(s.tolist()).decode().strip())
                except Exception:
                    sp_str.append(str(s).strip())
            else:
                sp_str.append(str(s).strip())
        try:
            idx = sp_str.index(param)
        except ValueError:
            return np.array([' '] * ds.sizes['N_PROF'])

        pdm = ds['PARAMETER_DATA_MODE'].values  # (N_PROF, N_PARAM)
        modes = np.array([
            (m.decode().strip() if isinstance(m, bytes) else str(m).strip())[:1] or ' '
            for m in pdm[:, idx]
        ])
        return modes

    if 'DATA_MODE' in ds:
        return decode_bytes(ds['DATA_MODE'].values)

    return np.array([' '] * ds.sizes['N_PROF'])


def add_colorbar_legend(fig, items, x_left, y_top, total_height,
                        title=None, block_width=0.025, label_offset=0.012):
    """Manual colorbar-style vertical legend at fixed paper x-position.

    items: top-to-bottom list of (label, color) tuples.
    x_left: paper-x of the LEFT edge of the color block column.
    y_top:  paper-y of the TOP edge.
    total_height: paper height of the entire color column.
    """
    n = len(items)
    block_h = total_height / n

    if title:
        fig.add_annotation(
            xref='paper', yref='paper',
            x=x_left + block_width / 2, y=y_top + 0.012,
            text=title, showarrow=False,
            xanchor='center', yanchor='bottom',
            font=dict(size=11, color='black'),
        )

    for i, (label, color) in enumerate(items):
        y0 = y_top - (i + 1) * block_h
        y1 = y_top - i * block_h
        fig.add_shape(
            type='rect', xref='paper', yref='paper',
            x0=x_left, x1=x_left + block_width,
            y0=y0, y1=y1,
            fillcolor=color,
            line=dict(width=0),
        )
        fig.add_annotation(
            xref='paper', yref='paper',
            x=x_left + block_width + label_offset,
            y=(y0 + y1) / 2,
            text=label, showarrow=False,
            xanchor='left', yanchor='middle',
            font=dict(size=11),
        )

    # Outer border for clean frame
    fig.add_shape(
        type='rect', xref='paper', yref='paper',
        x0=x_left, x1=x_left + block_width,
        y0=y_top - total_height, y1=y_top,
        fillcolor='rgba(0,0,0,0)',
        line=dict(width=0.6, color='#888'),
    )


def interp_to_grid(pres_2d, vals_2d, pres_grid):
    """Interpolate each profile onto a regular pressure grid.
       Returns (n_profiles, n_pres_grid) array, NaN outside profile range."""
    n_prof = pres_2d.shape[0]
    out = np.full((n_prof, len(pres_grid)), np.nan)
    for i in range(n_prof):
        p, v = pres_2d[i], vals_2d[i]
        m = ~np.isnan(p) & ~np.isnan(v)
        if m.sum() < 2:
            continue
        pv, vv = p[m], v[m]
        order = np.argsort(pv)
        pv, vv = pv[order], vv[order]
        _, ui = np.unique(pv, return_index=True)
        pv, vv = pv[ui], vv[ui]
        f = interp1d(pv, vv, bounds_error=False, fill_value=np.nan)
        out[i, :] = f(pres_grid)
    return out


# DAC code → (country, agency) — Argo Reference Table 4
DAC_INFO = {
    'AO': ('UNITED STATES',  'AOML'),
    'BO': ('UNITED KINGDOM', 'BODC'),
    'CS': ('AUSTRALIA',      'CSIRO'),
    'IF': ('FRANCE',         'Coriolis (Ifremer)'),
    'IN': ('INDIA',          'INCOIS'),
    'JA': ('JAPAN',          'JMA'),
    'KM': ('SOUTH KOREA',    'KMA'),
    'KO': ('SOUTH KOREA',    'KORDI'),
    'ME': ('CANADA',         'MEDS'),
    'NM': ('CHINA',          'NMDIS'),
}


def derive_country(dac_code):
    return DAC_INFO.get(dac_code, (dac_code, '?'))[0]


def derive_networks(meta):
    """Infer Argo programs from project + parameter list."""
    project = d(meta['PROJECT_NAME'].values).upper()
    params_list = [d(p) for p in meta['PARAMETER'].values]
    bgc_set = {'DOXY', 'CHLA', 'BBP700', 'NITRATE', 'PH_IN_SITU_TOTAL', 'CDOM'}
    is_bgc = bool(set(params_list) & bgc_set)
    nets = ['Argo', 'Global Core mission']
    nets.append('Argo BGC' if is_bgc else 'Argo Core')
    if 'GO-BGC' in project:
        nets.append('GO-BGC')
    if 'SOCCOM' in project:
        nets.append('SOCCOM')
    if 'EUROARGO' in project or 'EURO-ARGO' in project:
        nets.append('Euro-Argo')
    return nets


def derive_status(meta, prof):
    """3-tier: ACTIVE (≤30d) / INACTIVE (≤365d) / PRESUMED DEAD (>365d).
       If END_MISSION_DATE is set, status is CLOSED."""
    end_dt = parse_argo_date(meta['END_MISSION_DATE'].values)
    end_status = d(meta['END_MISSION_STATUS'].values)
    if end_dt is not None:
        if end_status == 'T':
            return 'CLOSED (no transmissions)'
        if end_status == 'R':
            return 'CLOSED (retrieved)'
        return 'CLOSED'
    last_dt = juld_to_dt(float(prof['JULD'].values[-1]))
    if last_dt is None:
        return 'UNKNOWN'
    days = (datetime.utcnow() - last_dt).days
    if days <= 30:
        return f'ACTIVE  (last profile {days} d ago)'
    if days <= 365:
        return f'INACTIVE  (last profile {days} d ago — may resume)'
    return f'PRESUMED DEAD  (silent for {days} d)'


def wigos_id(wmo):
    return f'0-22000-0-{wmo}'


def fmt_date(dt, with_ago=False):
    if dt is None:
        return 'n/a'
    s = dt.strftime('%Y-%m-%d %H:%M:%S')
    if with_ago:
        days = (datetime.utcnow() - dt).days
        if days >= 0:
            s += f'  ({days} days ago)'
    return s


def get_config(meta, key):
    """Look up a launch-config parameter by name. Returns float or None."""
    for n, v in zip(meta['LAUNCH_CONFIG_PARAMETER_NAME'].values,
                    meta['LAUNCH_CONFIG_PARAMETER_VALUE'].values):
        if d(n) == key:
            return float(v) if v < 99999 else None
    return None


def measured_cycle_time(prof):
    """Median spacing between consecutive profile timestamps, in hours."""
    juld = prof['JULD'].values
    juld = juld[~np.isnan(juld) & (juld < 999990)]
    if len(juld) < 2:
        return None
    diffs_hours = np.diff(np.sort(juld)) * 24
    return float(np.median(diffs_hours))


def cycle_time_line(meta, prof):
    cfg      = get_config(meta, 'CONFIG_DownTime_hours')
    measured = measured_cycle_time(prof)
    parts = []
    if cfg is not None:
        parts.append(f'configured = {cfg:.1f} h ({cfg/24:.1f} d)')
    if measured is not None:
        parts.append(f'measured ≈ {measured:.1f} h ({measured/24:.2f} d)')
    return ' | '.join(parts) if parts else 'n/a'


def last_cycle_surface_bottom(ds):
    """Shallowest & deepest valid (P, T, S) of the last cycle."""
    if ds is None:
        return None, None
    i = ds.sizes['N_PROF'] - 1
    pres_name = 'PRES_ADJUSTED' if 'PRES_ADJUSTED' in ds else 'PRES'
    temp_name = 'TEMP_ADJUSTED' if 'TEMP_ADJUSTED' in ds else 'TEMP'
    psal_name = 'PSAL_ADJUSTED' if 'PSAL_ADJUSTED' in ds else 'PSAL'
    pres = ds[pres_name].values[i, :]
    temp = ds[temp_name].values[i, :]
    psal = ds[psal_name].values[i, :]
    valid = (pres < 9999) & (temp < 9999) & (psal < 9999) & (pres > -9999)
    if not valid.any():
        return None, None
    p, t, s = pres[valid], temp[valid], psal[valid]
    surf = {'P': float(p[np.argmin(p)]), 'T': float(t[np.argmin(p)]),
            'S': float(s[np.argmin(p)])}
    bot  = {'P': float(p[np.argmax(p)]), 'T': float(t[np.argmax(p)]),
            'S': float(s[np.argmax(p)])}
    return surf, bot


# Color schemes
QC_COLOR_MAP = {
    1: '#2ca02c', 2: '#ffdd00', 3: '#ff8c00', 4: '#d62728',
    5: '#90ee90', 8: '#ff69b4', 9: '#bbbbbb', 0: '#dddddd',
}
QC_LEVELS = [0, 1, 2, 3, 4, 5, 8, 9]
DM_COLORS = {'D': '#1f77b4', 'A': '#ff7f0e', 'R': '#d62728', ' ': '#cccccc'}

# ── BATCH B: shared BGC parameter colors (used in Tab 3 single-cycle profiles
#    AND Tab 6 trajectory time series + per-cycle BGC profiles).
#    Keeping these consistent means "green = O₂" everywhere in the dashboard.
BGC_COLORS = {
    'DOXY':              C_GRN,       # green
    'CHLA':              '#2a9d8f',   # teal
    'BBP700':            C_PUR,       # purple
    'NITRATE':           C_ORG,       # orange
    'PH_IN_SITU_TOTAL':  C_RED,       # red
    'PPOX_DOXY':         C_GRN,       # green (paired with DOXY)
}
BGC_UNITS = {
    'DOXY':             'µmol/kg',
    'CHLA':             'mg/m³',
    'BBP700':           'm⁻¹',
    'NITRATE':          'µmol/kg',
    'PH_IN_SITU_TOTAL': '',
    'PPOX_DOXY':        'mbar',
}
BGC_LABELS = {
    'DOXY':             'O₂',
    'CHLA':             'Chl-a',
    'BBP700':           'BBP700',
    'NITRATE':          'NO₃⁻',
    'PH_IN_SITU_TOTAL': 'pH',
    'PPOX_DOXY':        'pO₂ (in-air)',
}

# Parameter catalog: (name, label, units, plotly-cmap, contour-step)
ALL_PARAMS = [
    ('TEMP',             'Temperature',     '°C',       'RdYlBu_r', 2.0),
    ('PSAL',             'Salinity',        'PSU',      'Viridis',  0.05),
    ('DOXY',             'Dissolved O₂',    'µmol/kg',  'Turbo',    None),
    ('CHLA',             'Chlorophyll-a',   'mg/m³',    'YlGn',     None),
    ('NITRATE',          'Nitrate',         'µmol/kg',  'Plasma',   None),
    ('PH_IN_SITU_TOTAL', 'pH',              '',         'RdYlBu_r', None),
    ('BBP700',           'BBP at 700 nm',   'm⁻¹',      'Magma',    None),
]


# ═══════════════════════════════════════════════════════════════════════════════
# ── BATCH A: TEXT-CARD RENDERERS ─────────────────────────────────────────────
# Each returns nothing, calls st.markdown / st.dataframe directly.
# ═══════════════════════════════════════════════════════════════════════════════

def render_main_information(meta, prof, sprof=None):
    if meta is None or prof is None:
        st.warning("meta.nc or prof.nc not loaded.")
        return
    wmo     = d(meta['PLATFORM_NUMBER'].values)
    dac     = d(meta['DATA_CENTRE'].values)
    country = derive_country(dac)
    model   = d(meta['PLATFORM_TYPE'].values)
    family  = d(meta['PLATFORM_FAMILY'].values)
    trans   = d(meta['TRANS_SYSTEM'].values[0])
    ptt     = d(meta['PTT'].values)
    ship    = d(meta['DEPLOYMENT_PLATFORM'].values)
    nets    = derive_networks(meta)
    status  = derive_status(meta, prof)

    ptt_part = f"   PTT: {ptt}" if ptt and ptt.lower() != 'n/a' else ''
    md = f"""
- **Reference**: `{wmo}`
- **WMO ID**: `{wmo}`
- **WIGOS ID**: `{wigos_id(wmo)}`
- **Status**: {status}
- **Country**: {country} ({dac})
- **Model**: {model} ({family.lower()})
- **Telecom**: {trans}{ptt_part}
- **Networks**: {", ".join(nets)}
- **Ship**: {ship}
"""
    st.markdown(md)


def render_tracking_lifecycle(meta, prof):
    if meta is None or prof is None:
        st.warning("meta.nc or prof.nc not loaded.")
        return
    launch_dt  = parse_argo_date(meta['LAUNCH_DATE'].values)
    launch_lat = float(meta['LAUNCH_LATITUDE'].values)
    launch_lon = float(meta['LAUNCH_LONGITUDE'].values)
    n          = prof.sizes['N_PROF']
    last_dt    = juld_to_dt(float(prof['JULD'].values[-1]))
    last_lat   = float(prof['LATITUDE'].values[-1])
    last_lon   = float(prof['LONGITUDE'].values[-1])
    last_cyc   = int(prof['CYCLE_NUMBER'].values[-1])

    md = f"""
**Deployed**
- Latitude: `{launch_lat:.4f}`
- Longitude: `{launch_lon:.4f}`
- Date: {fmt_date(launch_dt)}

**Latest observation**  ({n} profiles, latest = Cycle #{last_cyc})
- Latitude: `{last_lat:.4f}`
- Longitude: `{last_lon:.4f}`
- Date: {fmt_date(last_dt, with_ago=True)}
"""
    st.markdown(md)


def render_about_float(meta):
    if meta is None:
        st.warning("meta.nc not loaded.")
        return
    wmo      = d(meta['PLATFORM_NUMBER'].values)
    serial   = d(meta['FLOAT_SERIAL_NO'].values)
    maker    = d(meta['PLATFORM_MAKER'].values)
    ptype    = d(meta['PLATFORM_TYPE'].values)
    trans    = d(meta['TRANS_SYSTEM'].values[0])
    ptt      = d(meta['PTT'].values)
    owner    = d(meta['FLOAT_OWNER'].values)
    pi       = d(meta['PI_NAME'].values)
    dac      = d(meta['DATA_CENTRE'].values)
    dac_full = DAC_INFO.get(dac, (dac, dac))[1]
    op_inst  = d(meta['OPERATING_INSTITUTION'].values)

    md = f"""
- **WMO**: `{wmo}`
- **Serial number**: {serial}
- **Platform maker**: {maker}
- **Platform type**: {ptype}
- **Transmission**: {trans}
- **PTT**: {ptt if ptt else 'n/a'}
- **Owner**: {owner}
- **PI**: {pi}
- **Data Centre**: {dac_full} ({dac})
- **Operating institution**: {op_inst}

**Sensors:**
"""
    for s in [d(x) for x in meta['SENSOR'].values]:
        md += f"- {s}\n"
    st.markdown(md)


def render_deployment(meta):
    if meta is None:
        st.warning("meta.nc not loaded.")
        return
    launch_dt  = parse_argo_date(meta['LAUNCH_DATE'].values)
    startup_dt = parse_argo_date(meta['STARTUP_DATE'].values)
    start_dt   = parse_argo_date(meta['START_DATE'].values)
    launch_lat = float(meta['LAUNCH_LATITUDE'].values)
    launch_lon = float(meta['LAUNCH_LONGITUDE'].values)
    launch_qc  = d(meta['LAUNCH_QC'].values)
    ship       = d(meta['DEPLOYMENT_PLATFORM'].values)
    cruise     = d(meta['DEPLOYMENT_CRUISE_ID'].values)
    ref_st     = d(meta['DEPLOYMENT_REFERENCE_STATION_ID'].values)
    project    = d(meta['PROJECT_NAME'].values)
    pi         = d(meta['PI_NAME'].values)

    md = f"""
- **Launched**: {fmt_date(launch_dt, with_ago=True)}
- **Startup date**: {fmt_date(startup_dt)}
- **First dive**: {fmt_date(start_dt)}
- **Latitude**: `{launch_lat:.4f}`
- **Longitude**: `{launch_lon:.4f}`
- **Launch QC**: {launch_qc}
- **Ship**: {ship}
- **Cruise**: {cruise if cruise.lower() != 'n/a' else 'n/a'}
- **Reference station**: {ref_st if ref_st.lower() != 'n/a' else 'n/a'}
- **Project**: {project}
- **PI**: {pi}
"""
    st.markdown(md)


def render_cycle_activity(meta, prof, sprof):
    if meta is None or prof is None:
        st.warning("meta.nc or prof.nc not loaded.")
        return
    status    = derive_status(meta, prof)
    launch_dt = parse_argo_date(meta['LAUNCH_DATE'].values)
    last_dt   = juld_to_dt(float(prof['JULD'].values[-1]))
    last_cyc  = int(prof['CYCLE_NUMBER'].values[-1])
    n_profs   = prof.sizes['N_PROF']

    age_str = 'n/a'
    if launch_dt is not None and last_dt is not None:
        age_days = (last_dt - launch_dt).days
        age_str = f'{age_days/365.25:.2f} years  ({age_days} days)'

    # ── Data modes by parameter group ─────────────────────────────────────
    # Use Sprof if available (per-parameter modes), else fall back to prof DATA_MODE
    BGC_DISPLAY_NAMES = {
        'PH_IN_SITU_TOTAL': 'PH',
        'BBP700': 'Backscatter',
    }
    HIDE_PARAMS = {'CHLA_FLUORESCENCE'}  # redundant with CHLA

    ds_for_modes = sprof if sprof is not None else prof
    dm_lines = []
    if 'STATION_PARAMETERS' in ds_for_modes:
        from collections import defaultdict as _dd
        stat_params = decode_bytes(ds_for_modes['STATION_PARAMETERS'].values[0, :])
        # Group params by (R, A, D) signature
        param_counts = {}
        for p in stat_params:
            if not p or p in HIDE_PARAMS:
                continue
            modes_p = get_data_mode_per_param(ds_for_modes, p)
            r = int(np.sum(modes_p == 'R'))
            a = int(np.sum(modes_p == 'A'))
            d = int(np.sum(modes_p == 'D'))
            param_counts[p] = (r, a, d)

        sig_groups = _dd(list)
        for p, sig in param_counts.items():
            sig_groups[sig].append(p)

        # Sort: CTD core first, then by D-count descending
        CTD_CORE = {'PRES', 'TEMP', 'PSAL'}
        def _grp_key(item):
            sig, ps = item
            r, a, d = sig
            is_core = bool(CTD_CORE & set(ps))
            return (0 if is_core else 1, -d, ','.join(sorted(ps)))

        groups_sorted = sorted(sig_groups.items(), key=_grp_key)
        for sig, ps in groups_sorted:
            r, a, d = sig
            display_names = sorted(BGC_DISPLAY_NAMES.get(p, p) for p in ps)
            ps_str = ", ".join(display_names)
            dm_lines.append(f"  - {ps_str} — D={d}, A={a}, R={r}")
    else:
        # Fallback: just show overall DATA_MODE
        dm_arr = decode_bytes(prof['DATA_MODE'].values)
        n_R = int((dm_arr == 'R').sum())
        n_A = int((dm_arr == 'A').sum())
        n_D = int((dm_arr == 'D').sum())
        dm_lines.append(f"  - All parameters — D={n_D}, A={n_A}, R={n_R}")

    # ── Dive depth statistics ─────────────────────────────────────────────
    designed_depth = get_config(meta, 'CONFIG_ProfilePressure_dbar')
    pres_arr = (prof['PRES_ADJUSTED'].values if 'PRES_ADJUSTED' in prof
                else prof['PRES'].values)
    pres_clean = np.where(pres_arr > 99990, np.nan, pres_arr)
    max_per_cycle = np.nanmax(pres_clean, axis=1)
    valid_max = max_per_cycle[~np.isnan(max_per_cycle)]
    if len(valid_max) > 0:
        min_d = float(np.min(valid_max))
        max_d = float(np.max(valid_max))
        latest_d = float(valid_max[-1])
    else:
        min_d = max_d = latest_d = None

    surf, bot = last_cycle_surface_bottom(sprof if sprof is not None else prof)

    md = f"""
- **Status**: {status}
- **Age**: {age_str}
- **Last profile**: {fmt_date(last_dt, with_ago=True)}
- **Latest cycle**: #{last_cyc}  ({n_profs} profiles total)
- **Data modes**:
"""
    md += "\n".join(dm_lines) + "\n"

    md += "- **Dive depth**:\n"
    if designed_depth is not None:
        md += f"  - Designed: {designed_depth:.0f} dbar\n"
    if min_d is not None:
        md += f"  - Min: {min_d:.0f} dbar\n"
        md += f"  - Max: {max_d:.0f} dbar\n"
        md += f"  - Latest: {latest_d:.0f} dbar\n"

    if surf is not None:
        md += f"- **Last surface data**: {surf['P']:.2f} dbar, {surf['T']:.3f} °C, {surf['S']:.3f} PSU\n"
    if bot is not None:
        md += f"- **Last bottom data**: {bot['P']:.2f} dbar, {bot['T']:.3f} °C, {bot['S']:.3f} PSU\n"
    st.markdown(md)


def render_technical_details(meta, prof):
    if meta is None or prof is None:
        st.warning("meta.nc or prof.nc not loaded.")
        return
    battery  = d(meta['BATTERY_TYPE'].values)
    serial   = d(meta['FLOAT_SERIAL_NO'].values)
    firmware = d(meta['FIRMWARE_VERSION'].values)
    special  = d(meta['SPECIAL_FEATURES'].values)
    custom   = d(meta['CUSTOMISATION'].values)

    park_pres    = get_config(meta, 'CONFIG_ParkPressure_dbar')
    profile_pres = get_config(meta, 'CONFIG_ProfilePressure_dbar')
    ice_temp     = get_config(meta, 'CONFIG_IceDetection_degC')
    ice_active   = ice_temp is not None

    md = f"""
- **Battery**: {battery}
- **Serial number**: {serial}
- **Firmware**: {firmware}
- **Cycle time**: {cycle_time_line(meta, prof)}
- **Drift pressure**: {f'{park_pres:.0f} dbar' if park_pres else 'n/a'}
- **Profile pressure**: {f'{profile_pres:.0f} dbar' if profile_pres else 'n/a'}
- **Ice detection**: {f'Yes (threshold {ice_temp:.2f} °C)' if ice_active else 'No'}
"""
    if special and special.lower() != 'n/a':
        md += f"- **Special features**: {special[:200]}\n"
    if custom and custom.lower() != 'n/a':
        md += f"- **Customisation**: {custom[:200]}\n"
    st.markdown(md)

    cc = st.columns(2)
    with cc[0]:
        st.markdown("**Sensors (hardware):**")
        sensors_df = pd.DataFrame({
            'Sensor':     [d(x) for x in meta['SENSOR'].values],
            'Maker':      [d(x) for x in meta['SENSOR_MAKER'].values],
            'Model':      [d(x) for x in meta['SENSOR_MODEL'].values],
            'Serial No.': [d(x) for x in meta['SENSOR_SERIAL_NO'].values],
        })
        st.dataframe(sensors_df, use_container_width=True, hide_index=True)
    with cc[1]:
        st.markdown("**Sensor → Parameters:**")
        params_df = pd.DataFrame({
            'Parameter':  [d(x) for x in meta['PARAMETER'].values],
            'Sensor':     [d(x) for x in meta['PARAMETER_SENSOR'].values],
            'Units':      [d(x) for x in meta['PARAMETER_UNITS'].values],
        })
        sensor_params = (params_df.groupby('Sensor')['Parameter']
                         .apply(lambda s: ', '.join(s))
                         .reset_index()
                         .rename(columns={'Parameter': 'Parameters measured'}))
        st.dataframe(sensor_params, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ── TAB 2 EXPANDED RENDERERS (added in this iteration) ───────────────────────
# Drawn from notebook Cell B (meta.nc) — add depth to Float Configuration
# section beyond the basic text card.
# ═══════════════════════════════════════════════════════════════════════════════

def render_controller_and_transmission(meta):
    """Section A2 expander content: controller board + full transmission detail.
       These are usually 'n/a' for many fields, but operators want to confirm
       the float identity matches what's on record."""
    if meta is None:
        return

    # Controller board (firmware-running hardware)
    cb_pri_type   = d(meta['CONTROLLER_BOARD_TYPE_PRIMARY'].values)
    cb_pri_serial = d(meta['CONTROLLER_BOARD_SERIAL_NO_PRIMARY'].values)
    cb_sec_type   = d(meta['CONTROLLER_BOARD_TYPE_SECONDARY'].values)
    cb_sec_serial = d(meta['CONTROLLER_BOARD_SERIAL_NO_SECONDARY'].values)
    manual_ver    = d(meta['MANUAL_VERSION'].values)

    st.markdown("**Controller board**")
    st.markdown(
        f"""
- **Primary type**: {cb_pri_type or 'n/a'}
- **Primary serial**: {cb_pri_serial or 'n/a'}
- **Secondary type**: {cb_sec_type or 'n/a'}
- **Secondary serial**: {cb_sec_serial or 'n/a'}
- **Manual version**: {manual_ver or 'n/a'}
"""
    )

    # Transmission & positioning (TRANS_SYSTEM is multi-element on some floats)
    def _join_array(var):
        if var not in meta:
            return 'n/a'
        arr = meta[var].values
        if arr.ndim == 0:
            return d(arr) or 'n/a'
        items = [d(x) for x in arr.flat if d(x)]
        return ', '.join(items) if items else 'n/a'

    st.markdown("**Transmission & positioning**")
    st.markdown(
        f"""
- **Transmission system**: {_join_array('TRANS_SYSTEM')}
- **System ID**: {_join_array('TRANS_SYSTEM_ID')}
- **Frequency**: {_join_array('TRANS_FREQUENCY')}
- **PTT**: {d(meta['PTT'].values) or 'n/a'}
- **Positioning system**: {_join_array('POSITIONING_SYSTEM')}
"""
    )

    # Free-text fields — always show (operator confirms 'no anomaly')
    st.markdown("**Anomaly / customisation / special features**")
    st.markdown(
        f"""
- **Anomaly**: {d(meta['ANOMALY'].values) or '(none reported)'}
- **Special features**: {d(meta['SPECIAL_FEATURES'].values) or '(none)'}
- **Customisation**: {d(meta['CUSTOMISATION'].values) or '(none)'}
"""
    )


def render_parameter_specs(meta):
    """Merged parameter table:
       Parameter | Sensor | Units | Maker | Model | Serial No. | Accuracy | Resolution
       Ordered: Core (PRES/TEMP/PSAL) → BGC primary → DOXY auxiliaries → others.
       All values pulled directly from meta.nc — for-loop joins SENSOR table info
       (maker/model/serial) onto PARAMETER table by sensor name."""
    if meta is None or 'PARAMETER' not in meta:
        return

    # Custom display order; parameters not in this list go at the bottom (alphabetical)
    PARAM_ORDER = [
        'PRES', 'TEMP', 'PSAL',
        'DOXY', 'NITRATE', 'PH_IN_SITU_TOTAL', 'CHLA', 'BBP700',
        'TEMP_DOXY', 'PHASE_DELAY_DOXY', 'TEMP_VOLTAGE_DOXY',
    ]

    # Step 1: build sensor-name → (maker, model, serial) lookup
    sensor_lookup = {}
    n_sensor = meta.sizes.get('N_SENSOR', 0)
    for i in range(n_sensor):
        sname = d(meta['SENSOR'].values[i])
        sensor_lookup[sname] = {
            'Maker':      d(meta['SENSOR_MAKER'].values[i]),
            'Model':      d(meta['SENSOR_MODEL'].values[i]),
            'Serial No.': d(meta['SENSOR_SERIAL_NO'].values[i]),
        }

    # Step 2: walk PARAMETER list, join sensor info, collect rows
    n_param = meta.sizes['N_PARAM']
    rows = []
    for i in range(n_param):
        param  = d(meta['PARAMETER'].values[i])
        sensor = d(meta['PARAMETER_SENSOR'].values[i])
        sinfo  = sensor_lookup.get(sensor, {'Maker': 'n/a', 'Model': 'n/a', 'Serial No.': 'n/a'})
        rows.append({
            'Parameter':  param,
            'Sensor':     sensor,
            'Units':      d(meta['PARAMETER_UNITS'].values[i]),
            'Maker':      sinfo['Maker'],
            'Model':      sinfo['Model'],
            'Serial No.': sinfo['Serial No.'],
            'Accuracy':   d(meta['PARAMETER_ACCURACY'].values[i]),
            'Resolution': d(meta['PARAMETER_RESOLUTION'].values[i]),
        })

    # Step 3: sort by custom order; unknown params go last
    def sort_key(row):
        p = row['Parameter']
        return (PARAM_ORDER.index(p) if p in PARAM_ORDER else 999, p)
    rows.sort(key=sort_key)

    df = pd.DataFrame(rows)
    df = df.replace('', 'n/a')
    df = df[['Parameter', 'Sensor', 'Units', 'Maker', 'Model', 'Serial No.',
             'Accuracy', 'Resolution']]
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_predeployment_calibration(meta):
    """Factory calibration as a per-parameter list, ordered by PARAM_ORDER
       so it matches the Sensors / Physical parameters table above."""
    if meta is None or 'PARAMETER' not in meta:
        return

    PARAM_ORDER = [
        'PRES', 'TEMP', 'PSAL',
        'DOXY', 'NITRATE', 'PH_IN_SITU_TOTAL', 'CHLA', 'BBP700',
        'TEMP_DOXY', 'PHASE_DELAY_DOXY', 'TEMP_VOLTAGE_DOXY',
    ]

    n_param = meta.sizes['N_PARAM']
    indices = list(range(n_param))

    def sort_key(i):
        p = d(meta['PARAMETER'].values[i])
        return (PARAM_ORDER.index(p) if p in PARAM_ORDER else 999, p)
    indices.sort(key=sort_key)

    for i in indices:
        param = d(meta['PARAMETER'].values[i])
        eq    = d(meta['PREDEPLOYMENT_CALIB_EQUATION'].values[i])
        coef  = d(meta['PREDEPLOYMENT_CALIB_COEFFICIENT'].values[i])
        cmt   = d(meta['PREDEPLOYMENT_CALIB_COMMENT'].values[i])

        with st.expander(f"**{param}**", expanded=False):
            if eq and eq.upper() != 'N/A':
                st.markdown("**Equation**")
                st.code(eq, language='text')
            else:
                st.caption("No equation provided (CTD parameters typically have no equation).")
            if coef and coef.upper() != 'N/A' and coef.upper() != 'NA;':
                st.markdown("**Coefficients**")
                st.code(coef, language='text')
            if cmt and cmt.upper() != 'N/A':
                st.markdown("**Comment**")
                st.markdown(cmt)


# Plain-English meanings for Argo CONFIG_* parameter names. Used by both
# Launch configuration and Mission configuration tables.
CONFIG_MEANINGS = {
    'CONFIG_DownTime_hours':
        'Time below surface per cycle',
    'CONFIG_UpTime_hours':
        'Time at surface for transmission',
    'CONFIG_ParkPressure_dbar':
        'Drift depth (park pressure)',
    'CONFIG_ProfilePressure_dbar':
        'Maximum profile depth',
    'CONFIG_AscentToSurfaceTimeOut_hours':
        'Max ascent time before abort',
    'CONFIG_CPActivationPressure_dbar':
        'Continuous-profiling activation depth',
    'CONFIG_IceDetectionMixedLayerPMax_dbar':
        'Mixed-layer max depth used for ice detection',
    'CONFIG_IceDetectionMixedLayerPMin_dbar':
        'Mixed-layer min depth used for ice detection',
    'CONFIG_IceDetection_degC':
        'Temperature threshold for ice detection (abort ascent if colder)',
    'CONFIG_BitMaskMonthsIceDetectionActive_NUMBER':
        'Bitmask: months when ice detection is active (12 bits, one per month)',
    'CONFIG_MissionPreludeTime_hours':
        'Pre-mission settling time before first cycle',
    'CONFIG_ParkAndProfileCycleCounter_COUNT':
        'Park-and-profile cycle counter (1 = full cycle every time)',
    'CONFIG_BitMask_NUMBER':
        'Generic bitmask configuration',
    'CONFIG_ParkAndProfile_NUMBER':
        'Park-and-profile cycle pattern',
}


def _config_meaning(param_name):
    """Look up plain-English meaning, with prefix-match fallback."""
    if param_name in CONFIG_MEANINGS:
        return CONFIG_MEANINGS[param_name]
    # Fallback: strip trailing units suffix and try again
    for suffix in ('_dbar', '_hours', '_degC', '_NUMBER', '_COUNT', '_seconds'):
        if param_name.endswith(suffix):
            base = param_name[:-len(suffix)]
            for k, v in CONFIG_MEANINGS.items():
                if k.startswith(base):
                    return v
    return ''


def render_configuration_parameters(meta):
    """Combined launch + mission configuration.
       - Launch config: always shown as a 3-column table (Parameter | Value | Meaning)
       - Mission config: detect whether all missions are identical
            * if identical → 1-line text note (no redundant table)
            * if differ    → re-configuration history"""
    if meta is None:
        return

    # ───── Launch configuration ─────
    st.markdown("**Launch configuration**")
    st.caption("Parameters set at deployment and immutable thereafter.")

    if 'LAUNCH_CONFIG_PARAMETER_NAME' in meta:
        names  = meta['LAUNCH_CONFIG_PARAMETER_NAME'].values
        values = meta['LAUNCH_CONFIG_PARAMETER_VALUE'].values
        rows = []
        for n, v in zip(names, values):
            param = d(n)
            rows.append({
                'Parameter': param,
                'Value':     float(v) if v < 99999 else np.nan,
                'Meaning':   _config_meaning(param),
            })
        df_launch = pd.DataFrame(rows)
        st.dataframe(df_launch, use_container_width=True, hide_index=True)
    else:
        st.info("LAUNCH_CONFIG_PARAMETER_* not present in meta.nc.")

    st.markdown("")  # spacing
    st.markdown("**Mission configuration**")

    if 'CONFIG_PARAMETER_NAME' not in meta:
        st.info("CONFIG_PARAMETER_* not present in meta.nc.")
        return

    cfg_names    = meta['CONFIG_PARAMETER_NAME'].values
    cfg_values   = meta['CONFIG_PARAMETER_VALUE'].values
    mission_nums = meta['CONFIG_MISSION_NUMBER'].values
    n_missions   = len(mission_nums)

    # Wide table: rows = parameters, cols = missions
    cfg_table = pd.DataFrame(
        cfg_values.T,
        index=[d(n) for n in cfg_names],
        columns=[f'Mission {int(m)}' for m in mission_nums],
    )
    cfg_table = cfg_table.where(cfg_table < 99999, np.nan)

    all_identical = bool((cfg_table.nunique(axis=1, dropna=True) <= 1).all())

    if all_identical:
        st.markdown(
            f"This float has never been re-configured: all {n_missions} cycles use "
            "the same mission parameters (identical to launch configuration above)."
        )
    else:
        # Detect change events
        changed_rows = cfg_table.nunique(axis=1, dropna=True) > 1
        n_changed = int(changed_rows.sum())
        st.markdown(
            f"Float has been re-configured. **{n_changed} parameters** differ "
            f"across the {n_missions} mission records."
        )
        # Show the wide table
        st.dataframe(cfg_table.reset_index().rename(columns={'index': 'Parameter'}),
                     use_container_width=True, hide_index=True)
        if changed_rows.any():
            st.markdown("**Parameters that changed:**")
            for param in cfg_table.index[changed_rows]:
                vals = cfg_table.loc[param].dropna().unique()
                meaning = _config_meaning(param)
                m_part = f"  ({meaning})" if meaning else ""
                st.markdown(
                    f"- `{param}`{m_part}: " + " → ".join(f"{v:g}" for v in vals)
                )


def render_launch_configuration(meta):
    """Backward-compat wrapper kept in case other code paths call it.
       (The Tab 2 layout now uses render_configuration_parameters instead.)"""
    render_configuration_parameters(meta)


def render_mission_configuration(meta):
    """Backward-compat wrapper kept in case other code paths call it."""
    render_configuration_parameters(meta)


# ═══════════════════════════════════════════════════════════════════════════════
# ── BATCH A: VISUALIZATION BUILDERS (plotly) ─────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def make_section_plot(ds, param, label, units, cmap, contour_step):
    """Plotly section plot: depth–time heatmap + (optional) contours
       + DATA_MODE strip below + cycle-number ticks on top axis.
       Returns a plotly Figure or None if the parameter is missing."""

    vals = get_best(ds, param)
    if vals is None:
        return None

    qc = get_qc(ds, param)
    if qc is not None:
        keep = np.isin(qc, [1, 2, 5, 8])
        vals = np.where(keep, vals, np.nan)

    pres_2d = mask_fill(ds['PRES_ADJUSTED'].values if 'PRES_ADJUSTED' in ds
                        else ds['PRES'].values)
    if not np.isfinite(np.nanmax(pres_2d)):
        return None
    pres_max  = float(np.nanmax(pres_2d))
    pres_grid = np.arange(0, np.ceil(pres_max / 10) * 10 + 1, 5.0)

    grid = interp_to_grid(pres_2d, vals, pres_grid)  # (N_PROF, N_PRES)
    z = grid.T  # (N_PRES, N_PROF) for plotly: y=pres, x=date

    dates  = juld_to_dates(ds['JULD'].values)
    cycles = ds['CYCLE_NUMBER'].values.astype(int)
    modes  = get_data_mode_per_param(ds, param)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.92, 0.08],
        vertical_spacing=0.04,
    )

    # Heatmap
    units_part = f' ({units})' if units else ''
    fig.add_trace(go.Heatmap(
        x=dates, y=pres_grid, z=z,
        colorscale=cmap,
        colorbar=dict(title=f'{label}<br>{units_part}',
                      len=0.85, y=0.55, yanchor='middle', thickness=12),
        hovertemplate=('Date: %{x|%Y-%m-%d}<br>'
                       'Pressure: %{y:.0f} dbar<br>'
                       f'{label}: %{{z:.3f}}{units_part}<extra></extra>'),
        zsmooth=False,
    ), row=1, col=1)

    # Contour overlay (TEMP / PSAL only)
    if contour_step is not None:
        with np.errstate(invalid='ignore'):
            zmin = np.nanmin(z); zmax = np.nanmax(z)
        if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
            fig.add_trace(go.Contour(
                x=dates, y=pres_grid, z=z,
                contours=dict(
                    coloring='lines',
                    showlines=True,
                    showlabels=False,
                    start=float(np.floor(zmin / contour_step) * contour_step),
                    end=float(zmax),
                    size=float(contour_step),
                ),
                colorscale=[[0, 'rgba(0,0,0,1)'], [1, 'rgba(0,0,0,1)']],
                line=dict(width=0.5),
                showscale=False,
                hoverinfo='skip',
            ), row=1, col=1)

    # DATA_MODE strip
    mode_colors = [DM_COLORS.get(m, '#cccccc') for m in modes]
    mode_text = [f'Cycle {c} — DATA_MODE: {m}' for c, m in zip(cycles, modes)]
    fig.add_trace(go.Scatter(
        x=dates, y=[0] * len(dates),
        mode='markers',
        marker=dict(color=mode_colors, size=14, symbol='square'),
        text=mode_text,
        hovertemplate='%{text}<extra></extra>',
        showlegend=False,
    ), row=2, col=1)

    fig.update_yaxes(title_text='Pressure (dbar)', autorange='reversed',
                     row=1, col=1)
    fig.update_yaxes(visible=False, range=[-0.5, 0.5], row=2, col=1)

    # Top x-axis: cycle numbers (sparse)
    n_show = min(6, len(cycles))
    if n_show > 1:
        tick_idx = np.linspace(0, len(cycles) - 1, n_show, dtype=int)
        fig.update_xaxes(
            side='top',
            tickmode='array',
            tickvals=[dates[i] for i in tick_idx],
            ticktext=[str(cycles[i]) for i in tick_idx],
            title_text='Cycle number',
            row=1, col=1,
        )
    fig.update_xaxes(title_text='Date', row=2, col=1)

    fig.update_layout(
        title=f'{label}{units_part} — Section',
        height=540,
        margin=dict(t=70, b=50, l=70, r=140),
    )
    return fig


def make_overlay_grid(ds, params, ncols=3):
    """Plotly N-column grid of overlaid profiles (T-S diagram + parameters),
       lines colored by cycle number with a shared colorbar.
       ── MOD 2: ncols configurable (2 / 3 / 4)."""

    n_panels = len(params) + 1
    nrows = int(np.ceil(n_panels / ncols))

    titles = ['T–S Diagram'] + [f'Overlaid {p[0]}' for p in params]
    titles += [''] * (nrows * ncols - len(titles))

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=titles,
        # ── BATCH A FIX: more spacing so titles don't collide with axis labels
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )

    n_prof  = ds.sizes['N_PROF']
    cycles  = ds['CYCLE_NUMBER'].values.astype(int)
    cmin    = int(cycles.min()); cmax = int(cycles.max())
    pres_2d = mask_fill(ds['PRES_ADJUSTED'].values if 'PRES_ADJUSTED' in ds
                        else ds['PRES'].values)

    colors = pc.sample_colorscale('RdYlBu_r', np.linspace(0, 1, n_prof))

    # Panel (1,1): T-S diagram
    temp = get_best(ds, 'TEMP')
    psal = get_best(ds, 'PSAL')
    if temp is not None and psal is not None:
        for i in range(n_prof):
            m = ~np.isnan(temp[i]) & ~np.isnan(psal[i])
            if m.any():
                fig.add_trace(go.Scatter(
                    x=psal[i][m], y=temp[i][m],
                    mode='lines',
                    line=dict(width=0.7, color=colors[i]),
                    opacity=0.4,
                    showlegend=False,
                    hovertemplate=(f'Cycle {cycles[i]}<br>'
                                   'S=%{x:.3f}<br>T=%{y:.2f}°C<extra></extra>'),
                ), row=1, col=1)
    fig.update_xaxes(title_text='Salinity (PSU)', row=1, col=1)
    fig.update_yaxes(title_text='Temperature (°C)', row=1, col=1)

    # Other panels
    for k, (name, label, units, _, _) in enumerate(params, start=1):
        row = k // ncols + 1
        col = k % ncols + 1

        vals = get_best(ds, name)
        if vals is None:
            continue
        for i in range(n_prof):
            p = pres_2d[i]; v = vals[i]
            m = ~np.isnan(p) & ~np.isnan(v)
            if m.any():
                fig.add_trace(go.Scatter(
                    x=v[m], y=p[m],
                    mode='lines',
                    line=dict(width=0.7, color=colors[i]),
                    opacity=0.5,
                    showlegend=False,
                    hovertemplate=(f'Cycle {cycles[i]}<br>'
                                   f'{name}=%{{x:.4g}}<br>'
                                   'P=%{y:.0f} dbar<extra></extra>'),
                ), row=row, col=col)

        xlabel = f'{label} ({units})' if units else label
        fig.update_xaxes(title_text=xlabel, row=row, col=col)
        fig.update_yaxes(title_text='Pressure (dbar)', autorange='reversed',
                         row=row, col=col)

    # Phantom trace for cycle-number colorbar
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            color=[cmin, cmax],
            colorscale='RdYlBu_r',
            cmin=cmin, cmax=cmax,
            colorbar=dict(title='Cycle',
                          x=1.02, len=0.6, thickness=12),
            showscale=True,
        ),
        showlegend=False,
        hoverinfo='skip',
    ), row=1, col=1)

    fig.update_layout(
        # ── BATCH A FIX: taller rows so titles + axis labels both fit
        height=460 * nrows,
        margin=dict(t=40, r=170, b=60, l=70),
    )
    return fig


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=2)
def load_datasets(data_dir: str, wmo: str) -> dict:
    D = data_dir.rstrip(r"\/") + os.sep
    suffixes = {
        "prof":  "_prof.nc",
        "sprof": "_Sprof.nc",
        "meta":  "_meta.nc",   # ── BATCH A: load meta for text cards
        "tech":  "_tech.nc",
        "dtraj": "_Dtraj.nc",
        "rtraj": "_Rtraj.nc",
    }
    ds = {}
    for key, suffix in suffixes.items():
        path = D + wmo + suffix
        ds[key] = xr.open_dataset(path, decode_times=False) if os.path.exists(path) else None
    return ds


@st.cache_data(show_spinner=False, max_entries=2)
def parse_tech(data_dir: str, wmo: str) -> pd.DataFrame | None:
    """Parse tech.nc long format → tidy DataFrame with columns [cycle, param, value]."""
    path = data_dir.rstrip(r"\/") + os.sep + wmo + "_tech.nc"
    if not os.path.exists(path):
        return None
    tech = xr.open_dataset(path, decode_times=False)
    names  = decode_bytes(tech["TECHNICAL_PARAMETER_NAME"].values)
    values = decode_bytes(tech["TECHNICAL_PARAMETER_VALUE"].values)
    cycles = tech["CYCLE_NUMBER"].values.astype(int)

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

# ── BATCH A: top-level dataset references for cleaner code below
prof  = ds.get("prof")
sprof = ds.get("sprof")
meta  = ds.get("meta")

# Dataset to use for parameter / section visualizations:
#   - sprof if available (BGC float)
#   - prof  if not       (Core-only float)
DS_PARAMS = sprof if sprof is not None else prof
PARAMS = [p for p in ALL_PARAMS if (DS_PARAMS is not None and p[0] in DS_PARAMS)]


# ── Page header + top-level metrics ───────────────────────────────────────────
st.title(f"Float {WMO} — Monitoring Dashboard")

if prof is not None:
    n_prof = prof.sizes["N_PROF"]
    dates  = juld_to_dates(prof["JULD"].values)
    valid_d = [_d for _d in dates if pd.notna(_d)]

    # Float type detection
    has_bgc_h = sprof is not None
    designed_depth_h = get_config(meta, 'CONFIG_ProfilePressure_dbar') if meta is not None else None
    if designed_depth_h is not None and designed_depth_h > 2500:
        float_type_h = 'Deep'
    elif has_bgc_h:
        float_type_h = 'BGC'
    else:
        float_type_h = 'Core'

    # Actual mean dive depth from prof.nc max-per-cycle
    pres_h = (prof['PRES_ADJUSTED'].values if 'PRES_ADJUSTED' in prof
              else prof['PRES'].values)
    pres_h_clean = np.where(pres_h > 99990, np.nan, pres_h)
    max_h = np.nanmax(pres_h_clean, axis=1)
    valid_max_h = max_h[~np.isnan(max_h)]
    mean_depth_h = (f"{float(np.mean(valid_max_h)):.0f} dbar"
                    if len(valid_max_h) > 0 else 'n/a')
    designed_depth_str_h = (f"{designed_depth_h:.0f} dbar"
                            if designed_depth_h is not None else 'n/a')

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Float type",     float_type_h)
    c2.metric("Cycles",         n_prof)
    c3.metric("First cycle",    valid_d[0].strftime("%Y-%m-%d")  if valid_d else "—")
    c4.metric("Latest cycle",   valid_d[-1].strftime("%Y-%m-%d") if len(valid_d) > 1 else "—")
    c5.metric("Designed depth", designed_depth_str_h)
    c6.metric("Actual mean",    mean_depth_h)

st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# ── BATCH A: TABS ─────────────────────────────────────────────────────────────
# Tabs 1-3 implemented; Tabs 4-6 are placeholders for Batch B.
# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# TABS — 7 tabs total
# Order: overview → static info → dynamic engineering → data → QC → delivery → BGC
# ═══════════════════════════════════════════════════════════════════════════════
(tab_main,
 tab_meta,
 tab_health,
 tab_profiles,
 tab_qc,
 tab_delivery,
 tab_bgc) = st.tabs([
    "Main Information",
    "Float Metadata",
    "Float Health",
    "Profiles",
    "QC",
    "Data Delivery",
    "Trajectory data",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Main Information  [REDESIGNED: OceanOPS-style compact map]
# Layout: TOP ROW   = (Main Info + Tracking Lifecycle stacked)  ║  Map
#         BOTTOM    = 3 columns: About Float | Deployment | Cycle Activity
# ═══════════════════════════════════════════════════════════════════════════════
with tab_main:
    if prof is None:
        st.warning("prof.nc not found in the specified directory.")
    else:
        # ── Build the map figure (compact, global natural-earth view) ─────────
        lat    = mask_fill(prof["LATITUDE"].values)
        lon    = mask_fill(prof["LONGITUDE"].values)
        cycles_m = prof["CYCLE_NUMBER"].values.astype(int)
        dm_arr = decode_bytes(prof["DATA_MODE"].values)
        dates_m  = juld_to_dates(prof["JULD"].values)
        dates_s_m = [str(_d)[:10] if pd.notna(_d) else "—" for _d in dates_m]

        valid = ~np.isnan(lat) & ~np.isnan(lon)

        hover_text = [
            f"Cycle {c}<br>{ds_str}<br>({la:.3f}°, {lo:.3f}°)  [{md_}]"
            for c, ds_str, la, lo, md_ in zip(
                cycles_m[valid], np.array(dates_s_m)[valid],
                lat[valid], lon[valid], dm_arr[valid]
            )
        ]

        fig_map = go.Figure()
        fig_map.add_trace(go.Scattergeo(
            lon=lon[valid], lat=lat[valid], mode="lines",
            line=dict(width=1, color="rgba(0,119,182,0.4)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig_map.add_trace(go.Scattergeo(
            lon=lon[valid], lat=lat[valid], mode="markers",
            marker=dict(
                size=5, color=cycles_m[valid],
                colorscale="Plasma", showscale=True,
                colorbar=dict(title="Cycle", len=0.6, thickness=10,
                              x=1.02, xanchor="left"),
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="Profiles",
        ))

        v_idx = np.where(valid)[0]
        for idx, label_, sym, clr in [
            (v_idx[0],  "Start", "triangle-up",   "blue"),
            (v_idx[-1], "End",   "triangle-down", "red"),
        ]:
            fig_map.add_trace(go.Scattergeo(
                lon=[lon[idx]], lat=[lat[idx]],
                mode="markers+text",
                marker=dict(size=11, symbol=sym, color=clr),
                text=[f"{label_}"],
                textposition="top right",
                textfont=dict(size=10),
                name=f"{label_} ({dates_s_m[idx]})",
            ))

        # ── OceanOPS-style: GLOBAL natural-earth view, user can zoom in ──────
        # Don't restrict lataxis_range / lonaxis_range — show whole globe.
        # Box-zoom and scroll are enabled by default in plotly.
        fig_map.update_layout(
            geo=dict(
                projection_type="natural earth",
                showland=True,  landcolor="#f5f0eb",
                showocean=True, oceancolor="#cce5f5",
                showcoastlines=True, coastlinecolor="#999",
                showcountries=True, countrycolor="#bbb",
                domain=dict(x=[0, 1], y=[0, 1]),
            ),
            title=dict(
                text=f"Float {WMO} — {valid.sum()} position fixes  "
                     f"<span style='font-size:11px;color:#666'>"
                     f"(scroll/box-zoom to explore)</span>",
                font=dict(size=13),
            ),
            height=520,
            autosize=True,
            margin=dict(l=0, r=10, t=40, b=0),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.05,
                xanchor="center", x=0.5,
                font=dict(size=10),
            ),
        )

        # ── Map: full width ──────────────────────────────────────────────────
        st.plotly_chart(fig_map, use_container_width=True,
                        config={'responsive': True})

        # ── Top row: 2 cards (Main Info | Tracking Lifecycle) ────────────────
        cc1, cc2 = st.columns(2)
        with cc1:
            st.subheader("Main Information")
            render_main_information(meta, prof, sprof)
        with cc2:
            st.subheader("Tracking Lifecycle")
            render_tracking_lifecycle(meta, prof)

        st.divider()

        # ── Bottom row: 2 cards (Deployment | Cycle Activity) ────────────────
        cc3, cc4 = st.columns(2)
        with cc3:
            st.subheader("Deployment")
            render_deployment(meta)
        with cc4:
            st.subheader("Cycle Activity")
            render_cycle_activity(meta, prof, sprof)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Float Metadata  (meta.nc)
# Euro-Argo-style organization: short descriptive section names, mostly expanders.
# Top "About float" line is always visible (mirrors Tab 1 Main Information card).
# ═══════════════════════════════════════════════════════════════════════════════
with tab_meta:

    # ── Argo project information ─────────────────────────────────────────────
    # Merged: float identity (WMO/Status/etc.) + project information (PI/owner/etc.)
    with st.expander("Argo project information", expanded=False):
        st.caption("Float identity, project, PI, and operating institution.")
        if meta is not None and prof is not None:
            wmo     = d(meta['PLATFORM_NUMBER'].values)
            dac     = d(meta['DATA_CENTRE'].values)
            country = derive_country(dac)
            model   = d(meta['PLATFORM_TYPE'].values)
            family  = d(meta['PLATFORM_FAMILY'].values)
            trans   = d(meta['TRANS_SYSTEM'].values[0])
            ship    = d(meta['DEPLOYMENT_PLATFORM'].values)
            nets    = derive_networks(meta)
            status  = derive_status(meta, prof)
            dac_full = DAC_INFO.get(dac, (country, dac))[1]

            md = f"""
**Float identity**

- **Reference / WMO ID**: `{wmo}`
- **WIGOS ID**: `{wigos_id(wmo)}`
- **Status**: {status}
- **Country**: {country} ({dac})
- **Model**: {model} ({family.lower()})
- **Telecom**: {trans}
- **Networks**: {", ".join(nets)}
- **Ship**: {ship}

**Project**

- **Project name**: {d(meta['PROJECT_NAME'].values)}
- **PI**: {d(meta['PI_NAME'].values)}
- **Float owner**: {d(meta['FLOAT_OWNER'].values)}
- **Operating institution**: {d(meta['OPERATING_INSTITUTION'].values)}
- **Data centre**: {dac_full} ({dac})
"""
            st.markdown(md)

    # ── Platform information ─────────────────────────────────────────────────
    with st.expander("Platform information", expanded=False):
        st.caption("Hardware identity: battery, controller board, firmware, transmission.")
        render_controller_and_transmission(meta)

    # ── Deployment information ───────────────────────────────────────────────
    with st.expander("Deployment information", expanded=False):
        st.caption("When, where, and from what platform the float was deployed.")
        render_deployment(meta)

    # ── Sensors  (merged: hardware sensors + parameters + accuracy/resolution) ─
    with st.expander("Sensors", expanded=False):
        render_parameter_specs(meta)

    # ── Factory calibration ──────────────────────────────────────────────────
    with st.expander("Factory calibration", expanded=False):
        st.caption(
            "Calibration applied to each parameter at the factory, before deployment. "
            "These coefficients are used in the real-time pipeline for every cycle to "
            "convert raw sensor output into physical units."
        )
        render_predeployment_calibration(meta)

    # ── Configuration parameters  (merged: launch + mission) ─────────────────
    with st.expander("Configuration parameters", expanded=False):
        st.caption(
            "Launch configuration is set once at deployment; mission configuration is "
            "the active mission and can in principle be changed remotely."
        )
        render_configuration_parameters(meta)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Float Health  (tech.nc)
# Per-cycle engineering telemetry. Bullet summary at top, then expanders for
# each subsystem (Pressure / Buoyancy / Battery / Comm / Repositions / Ice /
# Piston / Status flags). Click "+" on each to expand.
# ═══════════════════════════════════════════════════════════════════════════════
with tab_health:

    if df_tech is None:
        st.warning("tech.nc not found — engineering telemetry unavailable.")
    else:
        # ── Pre-extract all parameters once (used across multiple sections) ──
        cyc_pump,   pump_time = get_param(df_tech, "TIME_BuoyancyPumpOn_seconds")
        cyc_volt,   voltage   = get_param(df_tech, "VOLTAGE_BatteryPumpOn_volts")
        cyc_bat,    current   = get_param(df_tech, "CURRENT_BatteryPumpOn_mA")
        cyc_vac,    vacuum    = get_param(df_tech, "PRESSURE_InternalVacuum_inHg")
        cyc_pres,   pres_off  = get_param(df_tech, "PRES_SurfaceOffsetNotTruncated_dbar")
        cyc_repos,  repos     = get_param(df_tech, "NUMBER_RepositionsDuringPark_COUNT")
        cyc_psurf,  p_surf    = get_param(df_tech, "POSITION_PistonSurface_COUNT")
        cyc_ppark,  p_park    = get_param(df_tech, "POSITION_PistonPark_COUNT")
        cyc_pnow,   p_now     = get_param(df_tech, "POSITION_PistonNow_COUNT")
        cyc_air,    air_blad  = get_param(df_tech, "PRESSURE_AirBladder_COUNT")
        cyc_descs,  descs     = get_param(df_tech, "NUMBER_PRESSamplesDuringDescentToPark_COUNT")
        cyc_gpst,   gps_t     = get_param(df_tech, "TIME_IridiumGPSFix_seconds")
        cyc_clk,    clock_dr  = get_param(df_tech, "CLOCK_RealTimeDrift_seconds")

        # Battery voltage panel inputs (4 series)
        cyc_vnl,    v_noload   = get_param(df_tech, "VOLTAGE_BatteryNoLoad_volts")
        cyc_vpiston,v_piston   = get_param(df_tech, "VOLTAGE_BatteryPistonPumpOn_volts")
        cyc_vsbe,   v_sbe      = get_param(df_tech, "VOLTAGE_BatterySBEPump_volts")

        # Battery current panel inputs (3 series)
        cyc_inl,    i_noload   = get_param(df_tech, "CURRENT_BatteryNoLoad_mA")
        cyc_isbe,   i_sbe      = get_param(df_tech, "CURRENT_BatterySBEPump_mA")

        # Pump-time linear trend
        if len(cyc_pump) > 5:
            slope_, intercept_, *_ = stats.linregress(cyc_pump, pump_time)
            pump_slope = slope_
            pump_trend = slope_ * cyc_pump + intercept_
        else:
            pump_slope = np.nan
            pump_trend = None

        # CTD status hex flag → integer
        ctd_rows = df_tech[df_tech["param"] == "FLAG_CTDStatus_hex"].copy()
        ctd_rows["int_val"] = ctd_rows["raw"].apply(
            lambda s: int(s, 16) if isinstance(s, str) and s else 0)
        ctd_int = ctd_rows["int_val"].values

        # ── HEALTH SUMMARY (grouped by expander section) ──────────────────────
        st.subheader("Health summary")

        # Compute signals
        n_ctd_bad      = int(np.sum(ctd_int > 0))
        n_repos_cycles = int(np.sum(repos >= 1)) if len(repos) else 0
        mean_pres_off  = float(np.nanmean(pres_off)) if len(pres_off) else np.nan
        peak_pres_off  = float(np.nanmax(np.abs(pres_off))) if len(pres_off) else np.nan
        vacuum_change  = (float(vacuum[-1] - vacuum[0])
                          if len(vacuum) > 1 else np.nan)

        # Air bladder slope
        if len(cyc_air) > 5:
            air_slope = float(stats.linregress(cyc_air, air_blad).slope)
        else:
            air_slope = None

        # Battery aging: NoLoad - PumpOn voltage gap, first 10 vs last 10 cycles
        bat_gap_early = bat_gap_late = None
        if len(cyc_vnl) > 0 and len(cyc_volt) > 0:
            common_b = np.intersect1d(cyc_vnl, cyc_volt)
            if len(common_b) >= 20:
                vnl_dict = dict(zip(cyc_vnl, v_noload))
                vbp_dict = dict(zip(cyc_volt, voltage))
                early = common_b[:10]
                late  = common_b[-10:]
                bat_gap_early = float(np.mean([vnl_dict[c] - vbp_dict[c] for c in early]))
                bat_gap_late  = float(np.mean([vnl_dict[c] - vbp_dict[c] for c in late]))

        # Pump current slope
        if len(cyc_bat) > 5:
            i_pump_slope = float(stats.linregress(cyc_bat, current).slope)
        else:
            i_pump_slope = None

        # GPS fix metrics
        gps_median = float(np.median(gps_t)) if len(gps_t) else np.nan
        gps_slope  = (float(stats.linregress(cyc_gpst, gps_t).slope)
                      if len(cyc_gpst) > 5 else None)

        # Clock drift
        clk_max = float(np.max(np.abs(clock_dr))) if len(clock_dr) else np.nan
        clk_sawtooth = False
        if len(clock_dr) > 5:
            clk_sawtooth = bool(np.any(clock_dr < -5)) and bool(np.any(clock_dr > -1))

        # Descent samples
        descs_mean = float(np.mean(descs)) if len(descs) else np.nan
        descs_std  = float(np.std(descs))  if len(descs) else np.nan

        # Ice
        ice_rows_h = df_tech[df_tech["param"] == "FLAG_IceDetected_bit"]
        if len(ice_rows_h) > 0:
            ice_int_h = ice_rows_h["value"].astype(float).fillna(0).astype(int).values
            n_ice_h = int(np.sum(ice_int_h > 0))
            n_ice_total_h = len(ice_int_h)
        else:
            n_ice_h = n_ice_total_h = 0

        # Piston gap
        common_p = np.intersect1d(cyc_psurf, cyc_ppark)
        if len(common_p) > 5:
            s_dict = dict(zip(cyc_psurf, p_surf))
            p_dict = dict(zip(cyc_ppark, p_park))
            gap_arr = np.array([s_dict[c] - p_dict[c] for c in common_p])
            piston_gap_mean  = float(np.mean(gap_arr))
            piston_gap_slope = float(stats.linregress(common_p, gap_arr).slope)
        else:
            piston_gap_mean = np.nan
            piston_gap_slope = None

        # Float status flag
        fs_rows_h = df_tech[df_tech["param"] == "FLAG_FloatStatus_hex"].copy()
        if len(fs_rows_h) > 0:
            fs_rows_h["int_val"] = fs_rows_h["raw"].apply(
                lambda s: int(s, 16) if isinstance(s, str) and s else 0)
            n_fs_bad_h = int(np.sum(fs_rows_h["int_val"].values > 0))
            fs_total_h = len(fs_rows_h)
        else:
            n_fs_bad_h = fs_total_h = 0

        # Build bullet list grouped by expander section
        md_lines = []

        md_lines.append("**Pressure**")
        if not np.isnan(peak_pres_off):
            if peak_pres_off > 20:
                s = f"peak ±{peak_pres_off:.2f} dbar — exceeds DMQC threshold"
            elif abs(mean_pres_off) > 5:
                s = f"mean {mean_pres_off:+.2f} dbar — review against ±20 dbar threshold"
            else:
                s = (f"mean {mean_pres_off:+.2f} dbar, peak ±{peak_pres_off:.2f} dbar — "
                     "well within DMQC tolerance")
            md_lines.append(f"- **Surface pressure offset** — {s}")
        if not np.isnan(vacuum_change):
            s = ("possible slow leak, monitor" if vacuum_change < -2.0 else "stable")
            md_lines.append(
                f"- **Internal vacuum** — {vacuum_change:+.2f} inHg over deployment — {s}"
            )
        if air_slope is not None:
            if air_slope < -0.3:
                s = "declining, pump or bladder may be degrading"
            elif air_slope > 0.3:
                s = "rising"
            else:
                s = "stable"
            md_lines.append(
                f"- **Air bladder pressure** — trend {air_slope:+.2f}/cycle — {s}"
            )

        md_lines.append("\n**Buoyancy**")
        if not np.isnan(pump_slope):
            s = "early buoyancy degradation signal" if pump_slope > 1.0 else "normal"
            md_lines.append(f"- **Pump time trend** — {pump_slope:+.2f} s/cycle — {s}")

        md_lines.append("\n**Battery**")
        if bat_gap_early is not None and bat_gap_late is not None:
            delta = bat_gap_late - bat_gap_early
            if abs(delta) > 0.3:
                s = (f"NoLoad–PumpOn voltage gap widened "
                     f"{bat_gap_early:.2f} → {bat_gap_late:.2f} V ({delta:+.2f}) — "
                     "internal resistance increasing")
            else:
                s = (f"NoLoad–PumpOn voltage gap stable "
                     f"({bat_gap_early:.2f} → {bat_gap_late:.2f} V)")
            md_lines.append(f"- **Battery aging** — {s}")
        if i_pump_slope is not None:
            if abs(i_pump_slope) < 0.5:
                s = f"current draw stable (slope {i_pump_slope:+.2f} mA/cycle)"
            elif i_pump_slope > 0.5:
                s = (f"current draw rising (slope {i_pump_slope:+.2f} mA/cycle) — "
                     "mechanical resistance increasing")
            else:
                s = f"current draw decreasing (slope {i_pump_slope:+.2f} mA/cycle)"
            md_lines.append(f"- **Pump current** — {s}")

        md_lines.append("\n**Communication & timing**")
        if not np.isnan(gps_median):
            extra = ""
            if gps_slope is not None:
                if gps_slope > 1:
                    extra = f", trend {gps_slope:+.2f} s/cycle — antenna may be degrading"
                else:
                    extra = f", trend {gps_slope:+.2f} s/cycle — stable"
            md_lines.append(f"- **GPS fix time** — median {gps_median:.0f} s{extra}")
        if not np.isnan(clk_max):
            if clk_sawtooth:
                s = f"max |drift| {clk_max:.0f} s, sawtooth pattern — normal (resets at GPS sync)"
            elif clk_max > 60:
                s = f"max |drift| {clk_max:.0f} s — large drift, investigate"
            else:
                s = f"max |drift| {clk_max:.0f} s"
            md_lines.append(f"- **Clock drift** — {s}")

        md_lines.append("\n**Repositions & drift**")
        if len(repos) > 0:
            if n_repos_cycles == 0:
                s = "0 cycles — velocity estimates clean"
            else:
                pct = n_repos_cycles / len(repos) * 100
                s = (f"{n_repos_cycles} of {len(repos)} cycles ({pct:.0f}%) — "
                     "velocity estimates contaminated")
            md_lines.append(f"- **Cycles with repositions** — {s}")
        if not np.isnan(descs_mean) and descs_mean > 0:
            cv = (descs_std / descs_mean * 100) if descs_mean else 0
            if cv > 50:
                s = (f"mean {descs_mean:.1f} samples, highly variable (cv={cv:.0f}%) — "
                     "check for sensor or comm issues")
            else:
                s = f"mean {descs_mean:.1f} samples — stable"
            md_lines.append(f"- **Pressure samples during descent** — {s}")

        md_lines.append("\n**Ice**")
        if n_ice_total_h == 0:
            md_lines.append("- **Ice detection** — no ice flag data in tech.nc")
        elif n_ice_h == 0:
            md_lines.append(
                f"- **Ice detection** — no ice events detected in {n_ice_total_h} cycles"
            )
        else:
            md_lines.append(
                f"- **Ice detection** — {n_ice_h} cycles "
                f"({n_ice_h/n_ice_total_h*100:.0f}%) had ice evasion triggered"
            )

        md_lines.append("\n**Piston**")
        if not np.isnan(piston_gap_mean) and piston_gap_slope is not None:
            if piston_gap_slope < -0.3:
                s = (f"mean gap {piston_gap_mean:.0f} counts, narrowing trend "
                     f"({piston_gap_slope:+.2f}/cycle) — piston may not fully extend")
            elif abs(piston_gap_slope) <= 0.3:
                s = f"mean gap {piston_gap_mean:.0f} counts, stable"
            else:
                s = (f"mean gap {piston_gap_mean:.0f} counts, widening "
                     f"({piston_gap_slope:+.2f}/cycle)")
            md_lines.append(f"- **Piston travel range (Surface − Park)** — {s}")

        md_lines.append("\n**Status flags (manufacturer-specific)**")
        if n_ctd_bad == 0:
            md_lines.append("- **CTD status flag** — all cycles clean")
        else:
            md_lines.append(
                f"- **CTD status flag anomalies** — {n_ctd_bad} cycles — "
                "cross-check against profile QC grades"
            )
        if fs_total_h > 0:
            if n_fs_bad_h == 0:
                md_lines.append("- **Float status flag** — all cycles clean")
            else:
                pct = n_fs_bad_h / fs_total_h * 100
                md_lines.append(
                    f"- **Float status flag** — non-zero on {n_fs_bad_h} of "
                    f"{fs_total_h} cycles ({pct:.0f}%) — manufacturer documentation "
                    "needed for interpretation"
                )

        st.markdown("\n".join(md_lines))

        st.divider()

        # =====================================================================
        # SECTION 1 — Pressure
        # =====================================================================
        with st.expander("Pressure", expanded=False):
            # Pressure offset (with ±20 dbar threshold)
            if len(cyc_pres) > 0:
                fig_po = go.Figure(go.Scatter(
                    x=cyc_pres, y=pres_off, mode="markers+lines",
                    marker=dict(size=4, color='#000000'), line=dict(width=1, color='#000000'),
                    hovertemplate="Cycle %{x}<br>%{y:+.2f} dbar<extra></extra>",
                ))
                for thr in (20, -20):
                    fig_po.add_hline(y=thr, line_dash="dot", line_color=C_RED, line_width=1.5)
                fig_po.add_annotation(
                    xref="paper", x=0.99, y=20, text="±20 dbar limit",
                    showarrow=False, font=dict(color=C_RED, size=10),
                    xanchor="right", yanchor="bottom",
                )
                fig_po.update_layout(
                    title="Surface pressure offset",
                    xaxis_title="Cycle number", yaxis_title="dbar",
                    height=300, margin=dict(t=50, b=50),
                    yaxis=dict(range=[-25, 25]),
                )
                st.plotly_chart(fig_po, use_container_width=True)
                st.caption(
                    "The pressure read by the CTD when the float is at the surface. "
                    "Should be near 0; deviation >±20 dbar (red dotted lines) triggers "
                    "DMQC pressure-bias adjustment. "
                    "NetCDF variable: `PRES_SurfaceOffsetNotTruncated_dbar`."
                )

            # Internal vacuum
            if len(cyc_vac) > 0:
                fig_iv = go.Figure(go.Scatter(
                    x=cyc_vac, y=vacuum, mode="markers+lines",
                    marker=dict(size=4, color='#000000'), line=dict(width=1, color='#000000'),
                    hovertemplate="Cycle %{x}<br>%{y:.2f} inHg<extra></extra>",
                ))
                fig_iv.update_layout(
                    title="Internal vacuum",
                    xaxis_title="Cycle number", yaxis_title="inHg",
                    height=300, margin=dict(t=50, b=50),
                )
                st.plotly_chart(fig_iv, use_container_width=True)
                st.caption(
                    "Vacuum maintained inside the float's pressure hull. "
                    "A steady decline indicates an O-ring slow leak — water is gradually "
                    "entering the hull. "
                    "NetCDF variable: `PRESSURE_InternalVacuum_inHg`."
                )

            # Air bladder
            if len(cyc_air) > 0:
                fig_ab = go.Figure(go.Scatter(
                    x=cyc_air, y=air_blad, mode="markers+lines",
                    marker=dict(size=4, color='#000000'), line=dict(width=1, color='#000000'),
                    hovertemplate="Cycle %{x}<br>%{y:.0f}<extra></extra>",
                ))
                fig_ab.update_layout(
                    title="Air bladder pressure",
                    xaxis_title="Cycle number", yaxis_title="count",
                    height=300, margin=dict(t=50, b=50),
                )
                st.plotly_chart(fig_ab, use_container_width=True)
                st.caption(
                    "Pressure in the external air bladder used to push the float to the "
                    "surface. Falling values mean the pump or bladder is degrading; "
                    "the float may struggle to surface. "
                    "NetCDF variable: `PRESSURE_AirBladder_COUNT`."
                )

        # =====================================================================
        # SECTION 2 — Buoyancy
        # =====================================================================
        with st.expander("Buoyancy", expanded=False):
            if len(cyc_pump) > 0:
                fig_pp = go.Figure()
                fig_pp.add_trace(go.Scatter(
                    x=cyc_pump, y=pump_time, mode="markers+lines",
                    marker=dict(size=4, color='#000000'), line=dict(width=1, color='#000000'),
                    name="Pump time",
                    hovertemplate="Cycle %{x}<br>%{y:.0f} s<extra></extra>",
                ))
                if pump_trend is not None:
                    fig_pp.add_trace(go.Scatter(
                        x=cyc_pump, y=pump_trend, mode="lines",
                        line=dict(width=2.5, color=C_RED, dash="dash"),
                        name=f"Trend  {pump_slope:+.2f} s/cycle",
                    ))
                fig_pp.update_layout(
                    title="Buoyancy pump run time",
                    xaxis_title="Cycle number", yaxis_title="seconds",
                    height=320, margin=dict(t=50, b=50),
                    legend=dict(x=0.01, y=0.99),
                )
                st.plotly_chart(fig_pp, use_container_width=True)
                st.caption(
                    "Seconds the buoyancy pump runs each cycle to push the float to the "
                    "surface. A gradual upward trend (red dashed line) indicates the "
                    "buoyancy system is working harder over time. "
                    "NetCDF variable: `TIME_BuoyancyPumpOn_seconds`."
                )

        # =====================================================================
        # SECTION 3 — Battery
        # =====================================================================
        with st.expander("Battery", expanded=False):
            # 4-voltage panel
            v_specs = [
                (cyc_vnl,     v_noload, "No load",          C_BLUE),
                (cyc_volt,    voltage,  "Buoyancy pump on", C_GRN),
                (cyc_vpiston, v_piston, "Piston pump on",   C_ORG),
                (cyc_vsbe,    v_sbe,    "SBE CTD pump on",  C_PUR),
            ]
            if any(len(c) > 0 for c, *_ in v_specs):
                fig_v = go.Figure()
                for cyc, vals, label, color in v_specs:
                    if len(cyc) == 0:
                        continue
                    fig_v.add_trace(go.Scatter(
                        x=cyc, y=vals, mode="markers+lines",
                        marker=dict(size=3, color=color), line=dict(width=1.2, color=color),
                        name=label,
                        hovertemplate=f"{label}<br>Cycle %{{x}}<br>%{{y:.2f}} V<extra></extra>",
                    ))
                fig_v.update_layout(
                    title="Battery voltage under different loads",
                    xaxis_title="Cycle number", yaxis_title="Volts",
                    height=380, margin=dict(t=50, b=80),
                    legend=dict(orientation="h", y=-0.18),
                )
                st.plotly_chart(fig_v, use_container_width=True)
                st.caption(
                    "Battery voltage measured during four different load conditions. "
                    "The gap between NoLoad voltage and loaded voltage equals the battery's "
                    "internal resistance — an increasing gap means the battery is aging."
                )

            # 3-current panel
            i_specs = [
                (cyc_inl,  i_noload, "No load",          C_BLUE),
                (cyc_bat,  current,  "Buoyancy pump on", C_GRN),
                (cyc_isbe, i_sbe,    "SBE CTD pump on",  C_PUR),
            ]
            if any(len(c) > 0 for c, *_ in i_specs):
                fig_i = go.Figure()
                for cyc, vals, label, color in i_specs:
                    if len(cyc) == 0:
                        continue
                    fig_i.add_trace(go.Scatter(
                        x=cyc, y=vals, mode="markers+lines",
                        marker=dict(size=3, color=color), line=dict(width=1.2, color=color),
                        name=label,
                        hovertemplate=f"{label}<br>Cycle %{{x}}<br>%{{y:.1f}} mA<extra></extra>",
                    ))
                fig_i.update_layout(
                    title="Battery current draw under different loads",
                    xaxis_title="Cycle number", yaxis_title="mA",
                    height=360, margin=dict(t=50, b=80),
                    legend=dict(orientation="h", y=-0.18),
                )
                st.plotly_chart(fig_i, use_container_width=True)
                st.caption(
                    "Current drawn by the battery under three load conditions. Rising current "
                    "under the same load indicates mechanical resistance is increasing "
                    "(pump or motor wear)."
                )

        # =====================================================================
        # SECTION 4 — Communication & timing
        # =====================================================================
        with st.expander("Communication & timing", expanded=False):
            if len(cyc_gpst) > 0:
                fig_gp = go.Figure(go.Scatter(
                    x=cyc_gpst, y=gps_t, mode="markers+lines",
                    marker=dict(size=4, color='#000000'), line=dict(width=1, color='#000000'),
                    hovertemplate="Cycle %{x}<br>%{y:.0f} s<extra></extra>",
                ))
                fig_gp.update_layout(
                    title="Time to acquire GPS fix",
                    xaxis_title="Cycle number", yaxis_title="seconds",
                    height=300, margin=dict(t=50, b=50),
                )
                st.plotly_chart(fig_gp, use_container_width=True)
                st.caption(
                    "Seconds taken to acquire a GPS fix at the surface. An upward trend "
                    "means the GPS antenna is degrading or satellite reception is harder "
                    "over time. Single-cycle spikes are usually transient. "
                    "NetCDF variable: `TIME_IridiumGPSFix_seconds`."
                )

            if len(cyc_clk) > 0:
                fig_ck = go.Figure(go.Scatter(
                    x=cyc_clk, y=clock_dr, mode="markers+lines",
                    marker=dict(size=4, color='#000000'), line=dict(width=1, color='#000000'),
                    hovertemplate="Cycle %{x}<br>%{y:+.0f} s<extra></extra>",
                ))
                fig_ck.update_layout(
                    title="Internal clock drift",
                    xaxis_title="Cycle number", yaxis_title="seconds",
                    height=280, margin=dict(t=50, b=50),
                )
                st.plotly_chart(fig_ck, use_container_width=True)
                st.caption(
                    "Drift of the float's internal clock relative to GPS-corrected time. "
                    "Sawtooth pattern is normal: drift accumulates between GPS syncs, then "
                    "resets to ~0 when the float gets a fix. Unbounded growth = clock failure. "
                    "NetCDF variable: `CLOCK_RealTimeDrift_seconds`."
                )

        # =====================================================================
        # SECTION 5 — Repositions & drift
        # =====================================================================
        with st.expander("Repositions & drift", expanded=False):
            if len(cyc_repos) > 0:
                bar_colors = [C_RED if r >= 1 else C_BLUE for r in repos]
                fig_rp = go.Figure(go.Bar(
                    x=cyc_repos, y=repos,
                    marker_color=bar_colors,
                    hovertemplate="Cycle %{x}<br>Repositions: %{y}<extra></extra>",
                ))
                fig_rp.add_hline(y=1, line_dash="dot", line_color=C_RED, line_width=1.5,
                                 annotation_text="velocity-QC threshold (≥1)",
                                 annotation_position="top right",
                                 annotation_font_color=C_RED)
                fig_rp.update_layout(
                    title="Repositions during park drift",
                    xaxis_title="Cycle number", yaxis_title="count",
                    height=320, margin=dict(t=50, b=50),
                )
                st.plotly_chart(fig_rp, use_container_width=True)
                st.caption(
                    "Number of times the float repositioned during its parking drift. "
                    "Argo derives parking-depth velocity from passive drift between "
                    "surface fixes; if the float repositioned (≥1, red bars), the velocity "
                    "estimate for that cycle is contaminated. "
                    "NetCDF variable: `NUMBER_RepositionsDuringPark_COUNT`."
                )
                n_bad = int(np.sum(repos >= 1))
                if n_bad > 0:
                    bad_cycles = sorted(cyc_repos[repos >= 1].tolist())
                    st.markdown(
                        f"**Cycles {bad_cycles[:20]}**"
                        + (" …" if n_bad > 20 else "")
                        + " — velocity estimates should be flagged for these cycles."
                    )

            if len(cyc_descs) > 0:
                fig_ds = go.Figure(go.Scatter(
                    x=cyc_descs, y=descs, mode="markers+lines",
                    marker=dict(size=4, color='#000000'), line=dict(width=1, color='#000000'),
                    hovertemplate="Cycle %{x}<br>%{y:.0f} samples<extra></extra>",
                ))
                fig_ds.update_layout(
                    title="Pressure samples during descent",
                    xaxis_title="Cycle number", yaxis_title="count",
                    height=280, margin=dict(t=50, b=50),
                )
                st.plotly_chart(fig_ds, use_container_width=True)
                st.caption(
                    "Number of pressure samples taken during descent to park depth. "
                    "Anomalously low values suggest a sensor or comm issue during descent; "
                    "constant values (typical) mean healthy descent. "
                    "NetCDF variable: `NUMBER_PRESSamplesDuringDescentToPark_COUNT`."
                )

        # =====================================================================
        # SECTION 6 — Ice
        # =====================================================================
        with st.expander("Ice", expanded=False):
            ice_rows = df_tech[df_tech["param"] == "FLAG_IceDetected_bit"].copy()
            if len(ice_rows) == 0:
                st.markdown("FLAG_IceDetected_bit not present in tech.nc.")
            else:
                ice_int = ice_rows["value"].astype(float).fillna(0).astype(int).values
                n_ice = int(np.sum(ice_int > 0))
                if n_ice == 0:
                    st.markdown(f"No ice events detected in {len(ice_rows)} cycles.")
                else:
                    st.markdown(f"**{n_ice} cycles** had ice evasion triggered.")
                    fig_ice = go.Figure(go.Bar(
                        x=ice_rows["cycle"], y=ice_int,
                        marker_color=[C_RED if v > 0 else C_BLUE for v in ice_int],
                    ))
                    fig_ice.update_layout(
                        title="Ice detection",
                        xaxis_title="Cycle number", yaxis_title="bit",
                        height=240, margin=dict(t=50, b=50),
                    )
                    st.plotly_chart(fig_ice, use_container_width=True)
                st.caption(
                    "Bitmask representing ice detection in the last 8 profiles. "
                    "Non-zero = ice was detected and the profile may have been aborted at "
                    "depth to avoid being crushed. Mainly relevant for floats in polar regions. "
                    "NetCDF variable: `FLAG_IceDetected_bit`."
                )

        # =====================================================================
        # SECTION 7 — Piston
        # =====================================================================
        with st.expander("Piston", expanded=False):
            if len(cyc_psurf) > 0 or len(cyc_ppark) > 0 or len(cyc_pnow) > 0:
                fig_pst = go.Figure()
                if len(cyc_pnow) > 0:
                    fig_pst.add_trace(go.Scatter(
                        x=cyc_pnow, y=p_now, mode="markers+lines",
                        marker=dict(size=4, color=C_BLUE), line=dict(width=1, color=C_BLUE),
                        name="Now (current cycle)",
                        hovertemplate="Cycle %{x}<br>Now: %{y:.0f}<extra></extra>",
                    ))
                if len(cyc_psurf) > 0:
                    fig_pst.add_trace(go.Scatter(
                        x=cyc_psurf, y=p_surf, mode="markers+lines",
                        marker=dict(size=4, color=C_GRN), line=dict(width=1, color=C_GRN),
                        name="At surface",
                        hovertemplate="Cycle %{x}<br>Surface: %{y:.0f}<extra></extra>",
                    ))
                if len(cyc_ppark) > 0:
                    fig_pst.add_trace(go.Scatter(
                        x=cyc_ppark, y=p_park, mode="markers+lines",
                        marker=dict(size=4, color=C_ORG), line=dict(width=1, color=C_ORG),
                        name="At park",
                        hovertemplate="Cycle %{x}<br>Park: %{y:.0f}<extra></extra>",
                    ))
                fig_pst.update_layout(
                    title="Piston positions",
                    xaxis_title="Cycle number", yaxis_title="Stepper count",
                    height=380, margin=dict(t=50, b=50),
                    legend=dict(orientation="h", y=-0.15),
                )
                st.plotly_chart(fig_pst, use_container_width=True)
                st.caption(
                    "Three piston states per cycle: current position (Now), position at "
                    "surface, position at park. The Surface − Park gap should remain wide; "
                    "if it narrows, the piston cannot fully extend and the float may fail "
                    "to surface."
                )

            # Nested: piston gap detail
            common = np.intersect1d(cyc_psurf, cyc_ppark)
            if len(common) > 0:
                s_vals = np.array([p_surf[cyc_psurf == c][0] for c in common])
                pp_vals = np.array([p_park[cyc_ppark == c][0] for c in common])
                gap = s_vals - pp_vals
                fig_gap = go.Figure(go.Scatter(
                    x=common, y=gap, mode="markers+lines",
                    marker=dict(size=4, color='#000000'), line=dict(width=1, color='#000000'),
                    hovertemplate="Cycle %{x}<br>gap: %{y:.0f}<extra></extra>",
                ))
                fig_gap.update_layout(
                    title="Piston travel range (Surface − Park)",
                    xaxis_title="Cycle number", yaxis_title="Stepper count gap",
                    height=300, margin=dict(t=50, b=50),
                )
                st.plotly_chart(fig_gap, use_container_width=True)
                st.caption(
                    "Surface position minus Park position, expressed as a single value. "
                    "A narrowing trend means the piston cannot fully extend → float may "
                    "fail to surface."
                )

        # =====================================================================
        # SECTION 8 — Status flags  (manufacturer-specific, kept for reference)
        # =====================================================================
        with st.expander("Status flags (manufacturer-specific)", expanded=False):
            st.caption(
                "Hex-encoded flags reported by the float firmware. "
                "Exact bit meaning is manufacturer-specific and not publicly documented; "
                "this section flags anomalous cycles for further investigation. "
                "When working a real fleet-monitoring case, cross-reference these against "
                "manufacturer documentation to interpret individual bits."
            )

            # CTD status
            st.markdown("**CTD status flag**")
            if n_ctd_bad > 0:
                st.markdown(f"{n_ctd_bad} non-zero cycles:")
                bad = ctd_rows[ctd_rows["int_val"] > 0]
                st.dataframe(
                    bad[["cycle", "raw", "int_val"]].rename(columns={
                        "cycle": "Cycle", "raw": "Hex flag", "int_val": "Integer value"
                    }),
                    use_container_width=True, hide_index=True,
                )
                st.caption(
                    "Non-zero CTD status = error condition from CTD firmware. "
                    "Cross-check against profile-level QC grades in prof.nc. "
                    "NetCDF variable: `FLAG_CTDStatus_hex`."
                )
            else:
                st.markdown("All cycles are 0 (no CTD errors reported).")
                st.caption("NetCDF variable: `FLAG_CTDStatus_hex`.")

            # Float status
            st.markdown("**Float status flag**")
            fs_rows = df_tech[df_tech["param"] == "FLAG_FloatStatus_hex"].copy()
            if len(fs_rows) > 0:
                fs_rows["int_val"] = fs_rows["raw"].apply(
                    lambda s: int(s, 16) if isinstance(s, str) and s else 0)
                n_fs_bad = int(np.sum(fs_rows["int_val"].values > 0))
                st.markdown(f"{n_fs_bad} non-zero / {len(fs_rows)} cycles")

                fig_fs = go.Figure(go.Bar(
                    x=fs_rows["cycle"], y=fs_rows["int_val"],
                    marker_color=[C_RED if v > 0 else C_GRN for v in fs_rows["int_val"]],
                    hovertemplate="Cycle %{x}<br>flag: %{y}<extra></extra>",
                ))
                fig_fs.update_layout(
                    title="Float status flag (per cycle)",
                    xaxis_title="Cycle number",
                    yaxis_title="Integer value of flag",
                    height=280, margin=dict(t=50, b=50),
                )
                st.plotly_chart(fig_fs, use_container_width=True)
                st.caption(
                    "Hex-encoded firmware status flag per cycle. Non-zero = firmware "
                    "reported some condition. Exact bit meaning is manufacturer-specific. "
                    "NetCDF variable: `FLAG_FloatStatus_hex`."
                )

                if n_fs_bad > 0:
                    bad_fs = fs_rows[fs_rows["int_val"] > 0]
                    bad_fs_show = bad_fs[["cycle", "raw", "int_val"]].rename(columns={
                        "cycle": "Cycle", "raw": "Hex flag", "int_val": "Integer value"
                    }).reset_index(drop=True)
                    st.dataframe(bad_fs_show, use_container_width=True, hide_index=True)
            else:
                st.markdown("FLAG_FloatStatus_hex not present in tech.nc.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Profiles & Sections
# Layout: 1) Section plots (per parameter)
#         2) All-profiles overlay grid (T-S + 7 params)
#         3) Single-cycle explorer (selectbox + per-cycle plots + calibration)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_profiles:
    if DS_PARAMS is None:
        st.warning("No profile file found (prof.nc / Sprof.nc).")
    else:
        # ──────────────────────────────────────────────────────────────────────
        # Section 1 — Section plots  [MOD 1: collapsible per-parameter]
        # ──────────────────────────────────────────────────────────────────────
        st.subheader("Section Plots")
        st.caption(
            "Good data only (QC flags 1, 2, 5, 8). "
            "Refer to QC tab for detailed QC flags."
        )
        if not PARAMS:
            st.info("No supported parameters found in this file.")
        for name, label, units, cmap, cstep in PARAMS:
            unit_part = f" ({units})" if units else ""
            with st.expander(f"{label}{unit_part}", expanded=False):
                with st.spinner(f"Building section plot for {name}…"):
                    fig_sec = make_section_plot(DS_PARAMS, name, label, units, cmap, cstep)
                if fig_sec is not None:
                    st.plotly_chart(fig_sec, use_container_width=True)
                else:
                    st.info(f"No data available for {name}.")

        st.divider()

        # ──────────────────────────────────────────────────────────────────────
        # Section 2 — All-profiles overlay grid  [MOD 2: ncols selector]
        # ──────────────────────────────────────────────────────────────────────
        st.subheader("All Profiles Overlay")
        if PARAMS:
            cc_layout = st.columns([1, 5])
            with cc_layout[0]:
                ncols_overlay = st.selectbox(
                    "Layout (columns)", options=[2, 3, 4],
                    index=1, key="overlay_ncols",
                )
            with st.spinner("Building overlay grid…"):
                fig_grid = make_overlay_grid(DS_PARAMS, PARAMS, ncols=ncols_overlay)
            st.plotly_chart(fig_grid, use_container_width=True)

        st.divider()

        # ──────────────────────────────────────────────────────────────────────
        # Section 3 — Single-cycle explorer (preserved from previous design)
        # ──────────────────────────────────────────────────────────────────────
        st.subheader("Single-cycle profile")

        # Prefer Sprof for this explorer too (gives BGC if available)
        prof_ds = DS_PARAMS

        n_prof_e = prof_ds.sizes["N_PROF"]
        cycles_e = prof_ds["CYCLE_NUMBER"].values.astype(int)
        if "DATA_MODE" in prof_ds:
            dm_e = decode_bytes(prof_ds["DATA_MODE"].values)
        elif "PARAMETER_DATA_MODE" in prof_ds:
            dm_e = decode_bytes(prof_ds["PARAMETER_DATA_MODE"].values[:, 0])
        else:
            dm_e = np.array(["?"] * n_prof_e)
        dates_e   = juld_to_dates(prof_ds["JULD"].values)
        dates_s_e = [str(_d)[:10] if pd.notna(_d) else "—" for _d in dates_e]

        is_sprof = sprof is not None
        has_bgc  = is_sprof and any(
            p in prof_ds for p in ["DOXY", "DOXY_ADJUSTED", "CHLA", "CHLA_ADJUSTED"]
        )

        cc_ctrl = st.columns([2, 5])
        with cc_ctrl[0]:
            sel_cycle = st.selectbox(
                "Select cycle",
                options=cycles_e.tolist(),
                format_func=lambda c: (
                    f"Cycle {c:3d}   "
                    f"{dates_s_e[np.where(cycles_e == c)[0][0]]}"
                ),
            )
        i_sel    = int(np.where(cycles_e == sel_cycle)[0][0])
        # Per-parameter modes for this cycle (used to annotate each subplot)
        # Falls back gracefully: if PARAMETER_DATA_MODE absent, all params share DATA_MODE
        def _mode_for_param(param_name):
            try:
                modes = get_data_mode_per_param(prof_ds, param_name)
                if modes is not None and len(modes) > i_sel:
                    return str(modes[i_sel])
            except Exception:
                pass
            return str(dm_e[i_sel])

        with cc_ctrl[1]:
            st.markdown(f"**Cycle {sel_cycle}**  ·  {dates_s_e[i_sel]}")

        def _get_var(name):
            if name in prof_ds:
                return mask_fill(prof_ds[name].values[i_sel])
            return None

        pres_raw  = _get_var("PRES")
        _pres_adj = _get_var("PRES_ADJUSTED")
        pres_adj  = _pres_adj if _pres_adj is not None else pres_raw
        temp_raw  = _get_var("TEMP")
        temp_adj  = _get_var("TEMP_ADJUSTED")
        psal_raw  = _get_var("PSAL")
        psal_adj  = _get_var("PSAL_ADJUSTED")

        def _valid_mask(x, p):
            if x is None or p is None:
                return None
            return ~np.isnan(x) & ~np.isnan(p)

        # Row 1a: T and S depth profiles  [MOD 3: cleaner styling, lighter raw lines]
        # Pattern: raw = light dotted line, adjusted = solid line + small markers
        # ── Row 1: T / S / T-S diagram in a single 1×3 row  [#3f]
        # Subplot titles include per-parameter DATA_MODE for this cycle
        _temp_mode = _mode_for_param('TEMP')
        _psal_mode = _mode_for_param('PSAL')
        fig_ts = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                f"Temperature (°C)  [{_temp_mode}]",
                f"Salinity (PSU)  [{_psal_mode}]",
                f"T-S Diagram  [{_temp_mode}/{_psal_mode}]",
            ],
            horizontal_spacing=0.08,
        )

        def _add_profile(fig, col, x_raw, x_adj, p_raw, p_adj, c_raw, c_adj, label_):
            m_r = _valid_mask(x_raw, p_raw)
            if m_r is not None and m_r.any():
                fig.add_trace(go.Scatter(
                    x=x_raw[m_r], y=p_raw[m_r], mode="lines",
                    line=dict(color=c_raw, width=1.2, dash="dot"),
                    name=f"{label_} raw",
                    hovertemplate=f"{label_} raw<br>x=%{{x:.3f}}<br>P=%{{y:.0f}} dbar<extra></extra>",
                ), row=1, col=col)
            m_a = _valid_mask(x_adj, p_adj)
            if m_a is not None and m_a.any():
                fig.add_trace(go.Scatter(
                    x=x_adj[m_a], y=p_adj[m_a], mode="lines+markers",
                    marker=dict(size=3, color=c_adj),
                    line=dict(color=c_adj, width=2),
                    name=f"{label_} adjusted",
                    hovertemplate=f"{label_} adjusted<br>x=%{{x:.3f}}<br>P=%{{y:.0f}} dbar<extra></extra>",
                ), row=1, col=col)

        _add_profile(fig_ts, 1, temp_raw, temp_adj, pres_raw, pres_adj,
                     "#a8c8e8", C_BLUE, "Temp")
        _add_profile(fig_ts, 2, psal_raw, psal_adj, pres_raw, pres_adj,
                     "#f4b8a0", C_RED, "Sal")

        # T-S diagram in column 3
        t_plot = temp_adj if temp_adj is not None else temp_raw
        s_plot = psal_adj if psal_adj is not None else psal_raw
        m_ts   = _valid_mask(t_plot, s_plot)
        if m_ts is not None and m_ts.any():
            p_col = pres_adj if pres_adj is not None else pres_raw
            fig_ts.add_trace(go.Scatter(
                x=s_plot[m_ts], y=t_plot[m_ts],
                mode="markers",
                marker=dict(
                    size=5, color=p_col[m_ts],
                    colorscale="Blues_r",
                    showscale=True,
                    colorbar=dict(
                        title="Pressure<br>(dbar)",
                        len=0.7, thickness=12,
                        tickformat=".0f",
                        x=1.02, xanchor='left',
                    ),
                    cmin=float(np.nanmin(p_col[m_ts])),
                    cmax=float(np.nanmax(p_col[m_ts])),
                ),
                showlegend=False,
                hovertemplate="S=%{x:.3f}  T=%{y:.2f}°C  P=%{marker.color:.0f} dbar<extra></extra>",
            ), row=1, col=3)
            fig_ts.update_xaxes(title_text="Salinity (PSU)", row=1, col=3)
            fig_ts.update_yaxes(title_text="Temperature (°C)", row=1, col=3)

        fig_ts.update_yaxes(autorange="reversed", title_text="Pressure (dbar)", col=1)
        fig_ts.update_yaxes(autorange="reversed", title_text="Pressure (dbar)", col=2)
        fig_ts.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
        fig_ts.update_xaxes(title_text="Salinity (PSU)", row=1, col=2)
        fig_ts.update_layout(
            height=460,
            showlegend=True,
            legend=dict(orientation="h", y=-0.18, font=dict(size=11)),
            margin=dict(b=80, t=70, r=120),
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Row 2: BGC parameters for this cycle (if available)
        if has_bgc:
            # ── BATCH B: derived from shared BGC_COLORS so Tab 3 ↔ Tab 6 match
            bgc_spec_order = ['DOXY', 'CHLA', 'NITRATE', 'PH_IN_SITU_TOTAL', 'BBP700']
            bgc_spec = []
            for base in bgc_spec_order:
                u = BGC_UNITS.get(base, '')
                label = BGC_LABELS.get(base, base)
                u_str = f"{label} ({u})" if u else label
                bgc_spec.append((f"{base}_ADJUSTED", u_str, BGC_COLORS[base]))
            avail_bgc = [(v, u, c) for v, u, c in bgc_spec
                         if v in prof_ds or v.replace("_ADJUSTED", "") in prof_ds]
            resolved = []
            for v, u, c in avail_bgc:
                actual = v if v in prof_ds else v.replace("_ADJUSTED", "")
                resolved.append((actual, u, c))

            if resolved:
                # Build subplot titles with per-parameter DATA_MODE for this cycle
                bgc_titles = []
                for actual_var, u, _ in resolved:
                    base_name = actual_var.replace("_ADJUSTED", "")
                    p_mode = _mode_for_param(base_name)
                    bgc_titles.append(f"{u}  [{p_mode}]")

                fig_row2 = make_subplots(
                    rows=1, cols=len(resolved),
                    subplot_titles=bgc_titles,
                    shared_yaxes=True,
                    horizontal_spacing=0.05,
                )
                for col_j, (var, unit, color) in enumerate(resolved, start=1):
                    vals = _get_var(var)
                    m = _valid_mask(vals, pres_adj)
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
                    showlegend=False,
                    margin=dict(b=60, t=50),
                )
                st.plotly_chart(fig_row2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — QC & Processing  [BATCH B]
# Layout: parameter selectbox → QC section heatmap + QC overlay scatter
#         + QC code reference + DATA_MODE reference
# ═══════════════════════════════════════════════════════════════════════════════
with tab_qc:
    if DS_PARAMS is None:
        st.warning("No profile file found (prof.nc / Sprof.nc).")
    elif not PARAMS:
        st.info("No supported parameters found in this file.")
    else:
        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  REFERENCE AREA (3 sub-sections, top-down)                       ║
        # ╚══════════════════════════════════════════════════════════════════╝

        # ── 1. DATA_MODE reference ────────────────────────────────────────────
        st.markdown("**DATA_MODE reference**")
        st.markdown(
            "| Code | Meaning |\n"
            "|---|---|\n"
            "| R | Real-time only — no adjustment applied |\n"
            "| A | Adjusted — automated correction, not expert-reviewed |\n"
            "| D | Delayed-mode — DMQC complete, use these for science |"
        )

        # ── 2. Data mode of all cycles ───────────────────────────────────────
        st.markdown("**Data mode of all cycles**")

        # Group parameters by (R_count, A_count, D_count) signature
        from collections import defaultdict as _dd

        # Determine which parameters this dataset has (use STATION_PARAMETERS row 0)
        if 'STATION_PARAMETERS' in DS_PARAMS:
            stat_params = decode_bytes(DS_PARAMS['STATION_PARAMETERS'].values[0, :])
        else:
            stat_params = np.array([])

        param_counts = {}
        param_modes = {}
        for p in stat_params:
            if not p:
                continue
            # get_data_mode_per_param() handles both Sprof (PARAMETER_DATA_MODE)
            # and prof (DATA_MODE) cases internally — returns 1D str array
            modes_p = get_data_mode_per_param(DS_PARAMS, p)
            r = int(np.sum(modes_p == 'R'))
            a = int(np.sum(modes_p == 'A'))
            d = int(np.sum(modes_p == 'D'))
            param_counts[p] = (r, a, d)
            param_modes[p] = modes_p

        # Group params with identical signatures
        sig_groups = _dd(list)
        for p, sig in param_counts.items():
            sig_groups[sig].append(p)

        # Sort groups: CTD core (PRES/TEMP/PSAL) first, then by D-count desc
        CTD_CORE = {'PRES', 'TEMP', 'PSAL'}
        def _grp_key(item):
            sig, ps = item
            r, a, d = sig
            is_core = bool(CTD_CORE & set(ps))
            return (0 if is_core else 1, -d, ','.join(sorted(ps)))
        groups_sorted = sorted(sig_groups.items(), key=_grp_key)

        # Plot one DATA_MODE strip per group  [#4e: heatmap for continuous look]
        cycles_qc = DS_PARAMS['CYCLE_NUMBER'].values.astype(int)
        DM_COLORS_TS = {'D': '#0077B6', 'A': '#f77f00', 'R': '#e63946'}
        # Discrete colorscale for D=0, A=1, R=2, missing=3
        _dm_idx = {'D': 0, 'A': 1, 'R': 2}
        _dm_colors_idx = ['#0077B6', '#f77f00', '#e63946', '#cccccc']
        _heatmap_scale = []
        for k, c in enumerate(_dm_colors_idx):
            _heatmap_scale.append([k / 4, c])
            _heatmap_scale.append([(k + 1) / 4, c])

        # cycle-number ticks for the last (bottom) figure  [#4f]
        n_show_ticks = min(6, len(cycles_qc))
        if n_show_ticks > 1:
            tick_idx = np.linspace(0, len(cycles_qc) - 1, n_show_ticks, dtype=int)
            tick_vals = [int(cycles_qc[i]) for i in tick_idx]
            tick_text = [str(int(cycles_qc[i])) for i in tick_idx]
        else:
            tick_vals = [int(c) for c in cycles_qc]
            tick_text = [str(int(c)) for c in cycles_qc]

        n_groups = len(groups_sorted)
        for i_grp, (sig, ps) in enumerate(groups_sorted):
            r, a, d = sig
            # [#4c] Always show R/A/D, even if 0
            counts_label = f"D={d}, A={a}, R={r}"
            params_label = ", ".join(sorted(ps))

            # Use first param's modes (all params in group have identical modes)
            mode_arr = param_modes[ps[0]]
            z_row = [[_dm_idx.get(m, 3) for m in mode_arr]]  # 1 row × N cycles

            hover_strip = [f'Cycle {c} — DATA_MODE={m}' for c, m in zip(cycles_qc, mode_arr)]

            is_last = (i_grp == n_groups - 1)

            fig_dm = go.Figure(go.Heatmap(
                x=cycles_qc, y=[0], z=z_row,
                colorscale=_heatmap_scale,
                zmin=-0.5, zmax=3.5,
                showscale=False,
                customdata=[hover_strip],
                hovertemplate='%{customdata}<extra></extra>',
                xgap=0, ygap=0,
            ))
            fig_dm.update_layout(
                height=85 if is_last else 70,
                margin=dict(t=32, b=(28 if is_last else 6), l=10, r=10),
                xaxis=dict(
                    title=('Cycle number' if is_last else None),
                    showticklabels=is_last,
                    tickmode='array' if is_last else 'auto',
                    tickvals=(tick_vals if is_last else None),
                    ticktext=(tick_text if is_last else None),
                ),
                yaxis=dict(visible=False, range=[-0.5, 0.5]),
                title=dict(
                    text=f'{params_label}  —  {counts_label}',
                    font=dict(size=14),  # [#4e] larger font
                    x=0.0, xanchor='left',
                ),
            )
            st.plotly_chart(fig_dm, use_container_width=True)

        st.divider()

        # ── 3. QC flag reference + Profile-level QC grade reference (side-by-side)
        cc_ref = st.columns(2)
        with cc_ref[0]:
            st.markdown("**QC flag reference**")
            st.markdown(
                "| Code | Meaning |\n"
                "|---|---|\n"
                "| 0 | No QC performed |\n"
                "| 1 | Good data |\n"
                "| 2 | Probably good |\n"
                "| 3 | Probably bad — correctable |\n"
                "| 4 | Bad — uncorrectable |\n"
                "| 5 | Value changed (DMQC) |\n"
                "| 8 | Estimated (interpolated / extrapolated) |\n"
                "| 9 | Missing value |"
            )
        with cc_ref[1]:
            st.markdown("**Profile-level QC grade**  (per parameter, per cycle)")
            st.markdown(
                "| Grade | Meaning |\n"
                "|---|---|\n"
                "| A | 100% of profile data is good (QC1) |\n"
                "| B | ≥75% good |\n"
                "| C | ≥50% good |\n"
                "| D | ≥25% good |\n"
                "| E | <25% good |\n"
                "| F | None of the data is good |"
            )

        st.divider()

        # ── INTRO ─────────────────────────────────────────────────────────────
        st.subheader("QC of every parameter")

        # ── shared resources for all parameters ───────────────────────────────
        pres_2d_q_all = mask_fill(
            DS_PARAMS['PRES_ADJUSTED'].values if 'PRES_ADJUSTED' in DS_PARAMS
            else DS_PARAMS['PRES'].values
        )
        dates_q_all  = juld_to_dates(DS_PARAMS['JULD'].values)
        cycles_q_all = DS_PARAMS['CYCLE_NUMBER'].values.astype(int)
        n_qc = len(QC_LEVELS)
        # discrete colorscale tied to QC_LEVELS
        discrete_scale = []
        for k, q in enumerate(QC_LEVELS):
            color = QC_COLOR_MAP[q]
            discrete_scale.append([k / n_qc,        color])
            discrete_scale.append([(k + 1) / n_qc,  color])
        code_to_idx = {q: k for k, q in enumerate(QC_LEVELS)}

        # ── colors for profile-level QC grades ────────────────────────────────
        # A = best (deep green) → F = worst (red); ' ' / missing = gray
        PROFILE_QC_COLORS = {
            'A': '#1a9850',  # deep green
            'B': '#91cf60',  # light green
            'C': '#fee08b',  # yellow
            'D': '#fc8d59',  # orange
            'E': '#d73027',  # red
            'F': '#7f0000',  # dark red
            ' ': '#dddddd',  # missing / unknown
            '':  '#dddddd',
        }

        # ── helper: try to fetch profile-level QC for a given param ───────────
        def get_profile_qc(ds, param):
            """Return per-cycle PROFILE_<param>_QC as length-N_PROF char array.
               Returns array of ' ' if not present in the dataset."""
            v = f'PROFILE_{param}_QC'
            if v in ds:
                return decode_bytes(ds[v].values)
            return np.array([' '] * ds.sizes['N_PROF'])

        # ── per-parameter expanders  [MOD 1-style "+" pattern] ────────────────
        for name, label, units, _, _ in PARAMS:
            unit_part = f" ({units})" if units else ""
            with st.expander(f"{label}{unit_part}", expanded=False):
                qc_arr_p = get_qc(DS_PARAMS, name)
                vals_p   = get_best(DS_PARAMS, name)

                if qc_arr_p is None:
                    st.info(f"No QC array available for {name}.")
                    continue

                # ── 1. QC SECTION CHART ───────────────────────────────────────
                st.markdown(f"**{label} — QC flags per depth & cycle**")

                pres_max_p = float(np.nanmax(pres_2d_q_all))
                if not np.isfinite(pres_max_p):
                    st.info(f"No valid pressure data.")
                    continue

                pres_grid_p = np.arange(0, np.ceil(pres_max_p / 10) * 10 + 1, 5.0)

                # interpolate QC onto grid
                grid_p = np.full((qc_arr_p.shape[0], len(pres_grid_p)), 9, dtype=np.int8)
                for i in range(qc_arr_p.shape[0]):
                    p = pres_2d_q_all[i]
                    m = ~np.isnan(p)
                    if m.sum() < 2:
                        continue
                    pv, qv = p[m], qc_arr_p[i][m]
                    order = np.argsort(pv)
                    pv, qv = pv[order], qv[order]
                    _, ui = np.unique(pv, return_index=True)
                    pv, qv = pv[ui], qv[ui]
                    f_intp = interp1d(pv, qv, kind='nearest',
                                      bounds_error=False, fill_value=9)
                    grid_p[i, :] = f_intp(pres_grid_p).astype(np.int8)

                grid_idx_p = np.vectorize(
                    lambda q: code_to_idx.get(int(q), n_qc - 1))(grid_p)

                modes_p = get_data_mode_per_param(DS_PARAMS, name)

                # ── MOD 5: profile-level QC strip alongside DATA_MODE strip
                profile_qc_p = get_profile_qc(DS_PARAMS, name)
                has_profile_qc = bool((profile_qc_p != ' ').any() and
                                      (profile_qc_p != '').any())

                if has_profile_qc:
                    fig_p = make_subplots(
                        rows=3, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.84, 0.08, 0.08],
                        vertical_spacing=0.04,
                    )
                else:
                    fig_p = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        row_heights=[0.92, 0.08],
                        vertical_spacing=0.04,
                    )

                # Heatmap (QC by depth × cycle)
                fig_p.add_trace(go.Heatmap(
                    x=dates_q_all, y=pres_grid_p, z=grid_idx_p.T,
                    colorscale=discrete_scale,
                    zmin=-0.5, zmax=n_qc - 0.5,
                    showscale=False,   # legend drawn manually below
                    hovertemplate=('Date: %{x|%Y-%m-%d}<br>'
                                   'Pressure: %{y:.0f} dbar<br>'
                                   'QC bin: %{z}<extra></extra>'),
                    zsmooth=False,
                ), row=1, col=1)

                # Profile-level QC strip (row 2 if available)
                if has_profile_qc:
                    pqc_colors = [PROFILE_QC_COLORS.get(g, '#dddddd')
                                  for g in profile_qc_p]
                    pqc_text = [f'Cycle {c} — Profile {name}_QC: {g}'
                                for c, g in zip(cycles_q_all, profile_qc_p)]
                    fig_p.add_trace(go.Scatter(
                        x=dates_q_all, y=[0] * len(dates_q_all),
                        mode='markers',
                        marker=dict(color=pqc_colors, size=14, symbol='square'),
                        text=pqc_text,
                        hovertemplate='%{text}<extra></extra>',
                        showlegend=False,
                    ), row=2, col=1)

                # DATA_MODE strip (last row)
                dm_row = 3 if has_profile_qc else 2
                mode_colors_p = [DM_COLORS.get(m, '#cccccc') for m in modes_p]
                mode_text_p = [f'Cycle {c} — DATA_MODE: {m}'
                               for c, m in zip(cycles_q_all, modes_p)]
                fig_p.add_trace(go.Scatter(
                    x=dates_q_all, y=[0] * len(dates_q_all),
                    mode='markers',
                    marker=dict(color=mode_colors_p, size=14, symbol='square'),
                    text=mode_text_p,
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=False,
                ), row=dm_row, col=1)

                fig_p.update_yaxes(title_text='Pressure (dbar)',
                                   autorange='reversed', row=1, col=1)
                if has_profile_qc:
                    fig_p.update_yaxes(visible=False, range=[-0.5, 0.5], row=2, col=1)
                    fig_p.update_yaxes(visible=False, range=[-0.5, 0.5], row=3, col=1)
                else:
                    fig_p.update_yaxes(visible=False, range=[-0.5, 0.5], row=2, col=1)

                # cycle-number top axis on row 1
                n_show = min(6, len(cycles_q_all))
                if n_show > 1:
                    tick_idx = np.linspace(0, len(cycles_q_all) - 1,
                                           n_show, dtype=int)
                    fig_p.update_xaxes(
                        side='top',
                        tickmode='array',
                        tickvals=[dates_q_all[i] for i in tick_idx],
                        ticktext=[str(cycles_q_all[i]) for i in tick_idx],
                        title_text='Cycle number',
                        row=1, col=1,
                    )
                fig_p.update_xaxes(title_text='Date', row=dm_row, col=1)

                # annotations for the strips (left labels)
                if has_profile_qc:
                    fig_p.add_annotation(
                        x=-0.01, y=0,
                        xref='paper',
                        yref='y2',
                        text='profile QC',
                        showarrow=False,
                        xanchor='right', yanchor='middle',
                        font=dict(size=10),
                    )
                fig_p.add_annotation(
                    x=-0.01, y=0,
                    xref='paper',
                    yref=f'y{dm_row}',
                    text='DATA_MODE',
                    showarrow=False,
                    xanchor='right', yanchor='middle',
                    font=dict(size=10),
                )

                # ── THREE VERTICAL LEGENDS (Cell 6 v4 design) ────────────────
                # All at same LEGEND_X. Each block has identical height (0.04).
                LEGEND_X = 1.015
                BLOCK_W = 0.025

                # QC flag — top
                add_colorbar_legend(
                    fig_p,
                    items=[(f'QC{q}', QC_COLOR_MAP[q]) for q in QC_LEVELS],
                    x_left=LEGEND_X,
                    y_top=0.97,
                    total_height=0.32,    # 8 levels × 0.04
                    title='<b>QC flag</b>',
                    block_width=BLOCK_W,
                )

                # Profile QC — middle (only if available)
                if has_profile_qc:
                    add_colorbar_legend(
                        fig_p,
                        items=[(g, PROFILE_QC_COLORS[g])
                               for g in ['A', 'B', 'C', 'D', 'E', 'F']],
                        x_left=LEGEND_X,
                        y_top=0.55,
                        total_height=0.24,    # 6 grades × 0.04
                        title='<b>Profile QC</b>',
                        block_width=BLOCK_W,
                    )

                # DATA_MODE — bottom
                add_colorbar_legend(
                    fig_p,
                    items=[(m, DM_COLORS[m]) for m in ['D', 'A', 'R']],
                    x_left=LEGEND_X,
                    y_top=0.20,
                    total_height=0.12,    # 3 modes × 0.04
                    title='<b>DATA_MODE</b>',
                    block_width=BLOCK_W,
                )

                fig_p.update_layout(
                    height=600 if has_profile_qc else 540,
                    margin=dict(t=70, b=50, l=100, r=140),
                )
                st.plotly_chart(fig_p, use_container_width=True)

                # ── 2. QC OVERLAY PROFILE  [MOD 7: all 8 levels with %] ──────
                st.markdown(f"**{label} — overlaid profiles colored by QC**")

                if vals_p is None:
                    st.info(f"No values array for {name}.")
                else:
                    pres_2d_o = pres_2d_q_all

                    # Total valid sample count (denominator for %)
                    valid_total = int(((qc_arr_p != 9)
                                       & ~np.isnan(vals_p)
                                       & ~np.isnan(pres_2d_o)).sum())
                    if valid_total == 0:
                        valid_total = 1   # prevent div-by-zero in label

                    fig_ov_p = go.Figure()
                    for q in QC_LEVELS:
                        m = ((qc_arr_p == q) & ~np.isnan(vals_p)
                             & ~np.isnan(pres_2d_o))
                        n_q = int(m.sum())
                        pct = n_q / valid_total * 100
                        if n_q > 0:
                            fig_ov_p.add_trace(go.Scattergl(
                                x=vals_p[m], y=pres_2d_o[m],
                                mode='markers',
                                marker=dict(size=3, color=QC_COLOR_MAP[q],
                                            opacity=0.65),
                                name=f'QC{q}  (n={n_q:,}, {pct:.1f}%)',
                                hovertemplate=(f'QC{q}<br>'
                                               'value=%{x:.4g}<br>'
                                               'P=%{y:.0f} dbar<extra></extra>'),
                            ))
                        else:
                            # MOD 7: show empty levels in legend too
                            fig_ov_p.add_trace(go.Scatter(
                                x=[None], y=[None],
                                mode='markers',
                                marker=dict(size=3, color=QC_COLOR_MAP[q],
                                            opacity=0.4),
                                name=f'QC{q}  (n=0, 0.0%)',
                                showlegend=True,
                                hoverinfo='skip',
                            ))

                    xlabel_p = f'{label} ({units})' if units else label
                    fig_ov_p.update_yaxes(autorange='reversed',
                                          title_text='Pressure (dbar)')
                    fig_ov_p.update_xaxes(title_text=xlabel_p)
                    fig_ov_p.update_layout(
                        height=520,
                        legend=dict(itemsizing='constant'),
                        margin=dict(t=30, l=70, r=20, b=60),
                    )
                    st.plotly_chart(fig_ov_p, use_container_width=True)

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  Scientific calibration records  (moved from Profiles tab)       ║
        # ╚══════════════════════════════════════════════════════════════════╝
        st.divider()
        st.subheader("Scientific calibration records")
        st.caption(
            "DMQC adjustments applied per cycle, per parameter. "
            "These records are written by the DMQC expert and document exactly "
            "what adjustment was applied. Select a cycle below."
        )

        if "SCIENTIFIC_CALIB_EQUATION" in DS_PARAMS:
            calib_cycles = DS_PARAMS['CYCLE_NUMBER'].values.astype(int)
            calib_cycle_options = [int(c) for c in calib_cycles]
            sel_cycle = st.selectbox(
                "Cycle number",
                options=calib_cycle_options,
                index=len(calib_cycle_options) - 1,
                key="qc_calib_cycle",
            )
            i_calib = int(np.where(calib_cycles == sel_cycle)[0][0])
            try:
                params_c = decode_bytes(DS_PARAMS["PARAMETER"].values[i_calib, 0, :])
                eq_c     = decode_bytes(DS_PARAMS["SCIENTIFIC_CALIB_EQUATION"].values[i_calib, 0, :])
                coeff_c  = decode_bytes(DS_PARAMS["SCIENTIFIC_CALIB_COEFFICIENT"].values[i_calib, 0, :])
                comm_c   = decode_bytes(DS_PARAMS["SCIENTIFIC_CALIB_COMMENT"].values[i_calib, 0, :])
                calib_df = pd.DataFrame({
                    "Parameter":   params_c,
                    "Equation":    eq_c,
                    "Coefficient": coeff_c,
                    "Comment":     comm_c,
                })
                calib_df = calib_df[calib_df["Parameter"].str.len() > 0].reset_index(drop=True)
                st.dataframe(calib_df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.caption(f"Could not extract calibration records: {e}")
        else:
            st.info("SCIENTIFIC_CALIB_* not present in this profile file.")

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  Subsurface velocity quality  (moved from Data Delivery tab)     ║
        # ╚══════════════════════════════════════════════════════════════════╝
        st.divider()
        st.subheader("Subsurface velocity quality")
        st.markdown(
            "Beyond profiles, subsurface ocean currents at parking depth (typically "
            "1000 dbar) can be estimated from passive Argo float drift between "
            "consecutive surface fixes. This estimate is contaminated when the float "
            "actively repositions during park (its buoyancy pump fires to maintain "
            "depth, e.g. in regions of strong vertical motion or fronts). Argo's data "
            "does not include ocean currents nor an associated QC flag, so estimating "
            "the quality of u, v requires counting repositions in `tech.nc`."
        )

        if df_tech is None:
            st.info("tech.nc not available — reposition counts unknown.")
        else:
            cyc_rep, rep_vals = get_param(df_tech, "NUMBER_RepositionsDuringPark_COUNT")
            if len(cyc_rep) == 0:
                st.info("No reposition records found in tech.nc.")
            else:
                max_repos = int(np.nanmax(rep_vals)) if len(rep_vals) else 1
                slider_max = max(20, max_repos)
                threshold = st.slider(
                    "User can adjust the reposition threshold. "
                    "It will flag cycles with ≥ N repositions.",
                    min_value=1, max_value=slider_max, value=1, step=1,
                    help=("Strict (1) flags any active control by the float. "
                          "Loose values (5+) flag only egregious cases."),
                )

                flagged_mask = rep_vals >= threshold
                flagged_cycles = sorted(cyc_rep[flagged_mask].tolist())
                n_flagged = len(flagged_cycles)
                nonzero_repos = rep_vals[rep_vals > 0]
                median_when_nonzero = (float(np.median(nonzero_repos))
                                       if len(nonzero_repos) else 0.0)

                # Chart first
                fig_rep = go.Figure()
                blue_mask = rep_vals < threshold
                red_mask  = rep_vals >= threshold
                if blue_mask.any():
                    fig_rep.add_trace(go.Bar(
                        x=cyc_rep[blue_mask], y=rep_vals[blue_mask],
                        marker_color=C_BLUE,
                        name='Below threshold',
                        hovertemplate="Cycle %{x}<br>Repositions: %{y}<extra></extra>",
                    ))
                if red_mask.any():
                    fig_rep.add_trace(go.Bar(
                        x=cyc_rep[red_mask], y=rep_vals[red_mask],
                        marker_color=C_RED,
                        name='Above threshold (velocity estimate flagged)',
                        hovertemplate="Cycle %{x}<br>Repositions: %{y}<extra></extra>",
                    ))
                fig_rep.add_hline(
                    y=threshold, line_dash="dot", line_color=C_RED, line_width=1.5,
                    annotation_text=f"threshold (≥{threshold})",
                    annotation_position="top right",
                    annotation_font_color=C_RED,
                )
                fig_rep.update_layout(
                    xaxis_title="Cycle number",
                    yaxis_title="Reposition count",
                    height=340,
                    showlegend=True,
                    legend=dict(orientation='h', y=-0.22),
                    margin=dict(t=30, b=80),
                )
                st.plotly_chart(fig_rep, use_container_width=True)

                # Diagnosis bullets AFTER the chart
                plural = "s" if threshold > 1 else ""
                cycles_str = str(flagged_cycles[:20])
                if n_flagged > 20:
                    cycles_str = cycles_str.rstrip(']') + ', …]'

                st.markdown(
                    f"**Diagnosis for this float (threshold = ≥ {threshold} reposition{plural}):**\n"
                    f"- {n_flagged} of {len(cyc_rep)} cycles flagged\n"
                    f"- Cycles flagged: {cycles_str}\n"
                    f"- Median repositions (when nonzero): {median_when_nonzero:.0f}\n"
                    f"- Max repositions in a single cycle: {max_repos}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Data Delivery  [BATCH B]
# Layout: 1) Variables and parameters table
#         2) Delivery performance (4 metrics + delay scatter on date axis)
#         3) DM eligibility table
#         4) Velocity QC reposition count (preserved from prior version)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_delivery:
    # ── 1. Variables and parameters table ─────────────────────────────────────
    st.subheader("Variables and parameters")

    if DS_PARAMS is None or not PARAMS:
        st.info("No parameters available — profile file missing.")
    else:
        dates_v = np.array(juld_to_dates(DS_PARAMS['JULD'].values), dtype=object)
        rows_v = []
        for name, label, units, *_ in PARAMS:
            vals_v = get_best(DS_PARAMS, name)
            if vals_v is None:
                continue
            valid_per_prof = ~np.all(np.isnan(vals_v), axis=1)
            if valid_per_prof.any():
                first_idx = int(np.argmax(valid_per_prof))
                last_idx  = len(valid_per_prof) - 1 - int(np.argmax(valid_per_prof[::-1]))
                first_str = (dates_v[first_idx].strftime('%Y-%m-%d')
                             if dates_v[first_idx] is not pd.NaT else 'n/a')
                last_str  = (dates_v[last_idx].strftime('%Y-%m-%d')
                             if dates_v[last_idx]  is not pd.NaT else 'n/a')
            else:
                first_str = last_str = 'n/a'
            rows_v.append({
                'Variable':        label,
                'Argo name':       name,
                'P02 reference':   f'SDN:P02::{name}',
                'Units':           units,
                'First obs':       first_str,
                'Latest obs':      last_str,
                'N profiles w/ data': int(valid_per_prof.sum()),
            })
        # [#5a] hover tooltips on column headers
        st.dataframe(
            pd.DataFrame(rows_v),
            use_container_width=True, hide_index=True,
            column_config={
                'Variable': st.column_config.TextColumn(
                    'Variable',
                    help='Each row is a measured parameter on this float.',
                ),
                'Argo name': st.column_config.TextColumn(
                    'Argo name',
                    help="Argo's standard variable name (used in NetCDF files).",
                ),
                'P02 reference': st.column_config.TextColumn(
                    'P02 reference',
                    help=("SeaDataNet's vocabulary mapping each parameter to a "
                          "standard ID. Lets different ocean databases "
                          "(Argo / GO-SHIP / OceanSITES / gliders) reference the "
                          "same physical measurement using a common identifier."),
                ),
                'Units': st.column_config.TextColumn('Units'),
                'First obs': st.column_config.TextColumn(
                    'First obs', help='First cycle date with valid data for this parameter.',
                ),
                'Latest obs': st.column_config.TextColumn(
                    'Latest obs', help='Most recent cycle date with valid data.',
                ),
                'N profiles w/ data': st.column_config.NumberColumn(
                    'N profiles w/ data',
                    help='Number of cycles with non-fill values for this parameter.',
                ),
            },
        )

    st.divider()

    # ── 2. Delivery performance ──────────────────────────────────────────────
    # [#5b] new section title
    st.subheader("How long does the data take to be delivered?")
    # [#5c] removed "The plot below shows..." sentence
    # [#5d] 12 hours not bold
    st.markdown(
        "After ascent, each cycle the float surfaces and transmits its data via Iridium "
        "satellite to the ground station, then onward to the GDAC. Argo's real-time "
        "target is to have data available at the GDAC within 12 hours of surfacing."
    )

    traj_d = ds.get("dtraj") if ds.get("dtraj") is not None else ds.get("rtraj")
    if traj_d is None:
        st.info("No trajectory file (Dtraj.nc / Rtraj.nc) — cannot compute delays.")
    else:
        TARGET_H = 12.0
        asc_end_all  = mask_fill(traj_d['JULD_ASCENT_END'].values)
        tx_start_all = mask_fill(traj_d['JULD_TRANSMISSION_START'].values)

        valid_t = ~np.isnan(asc_end_all) & ~np.isnan(tx_start_all)
        delay_h_all = (tx_start_all[valid_t] - asc_end_all[valid_t]) * 24.0
        asc_dates_all = np.array([JREF + pd.Timedelta(days=float(j))
                                  for j in asc_end_all[valid_t]])

        plaus = (delay_h_all >= 0) & (delay_h_all < 120)
        delay_h_v   = delay_h_all[plaus]
        asc_dates_v = asc_dates_all[plaus]

        if len(delay_h_v) == 0:
            st.info(
                "Transmission timing variables (JULD_ASCENT_END, JULD_TRANSMISSION_START) "
                "are not populated. This is common for older Argos floats."
            )
        else:
            on_d  = delay_h_v <= TARGET_H
            off_d = ~on_d
            n_on_d, n_off_d = int(on_d.sum()), int(off_d.sum())
            pct_on_d = n_on_d / len(delay_h_v) * 100

            cc_d = st.columns(4)
            cc_d[0].metric("Cycles plotted", len(delay_h_v))
            cc_d[1].metric("On-time (≤ 12 h)", f"{pct_on_d:.0f}%",
                           delta_color="normal" if pct_on_d >= 90 else "inverse")
            cc_d[2].metric("Median delay", f"{np.median(delay_h_v):.1f} h")
            cc_d[3].metric("Max delay",    f"{np.max(delay_h_v):.1f} h")

            fig_del = go.Figure()
            fig_del.add_trace(go.Scatter(
                x=asc_dates_v[on_d], y=delay_h_v[on_d],
                mode='markers',
                marker=dict(size=7, color=C_GRN, opacity=0.85),
                name=f'On target ≤ {TARGET_H:.0f} h  (n={n_on_d})',
                hovertemplate='%{x|%Y-%m-%d}<br>%{y:.1f} h<extra></extra>',
            ))
            fig_del.add_trace(go.Scatter(
                x=asc_dates_v[off_d], y=delay_h_v[off_d],
                mode='markers',
                marker=dict(size=8, color=C_RED, opacity=0.9),
                name=f'Off target > {TARGET_H:.0f} h  (n={n_off_d})',
                hovertemplate='%{x|%Y-%m-%d}<br>%{y:.1f} h<extra></extra>',
            ))
            fig_del.add_hline(
                y=TARGET_H, line_dash='dash', line_color=C_RED, line_width=1,
                annotation_text=f'{TARGET_H:.0f}-h target',
                annotation_position='top right',
                annotation_font_color=C_RED,
            )
            fig_del.update_layout(
                title="Transmission delay per cycle",
                xaxis_title="Surface date",
                yaxis_title="Delay (hours)",
                height=440,
                hovermode='closest',
            )
            st.plotly_chart(fig_del, use_container_width=True)

    st.divider()

    # ── 3. DM eligibility ────────────────────────────────────────────────────
    st.subheader("Delayed-mode eligibility")

    if DS_PARAMS is None or prof is None or not PARAMS:
        st.info("DM eligibility requires both a profile file and parameter list.")
    else:
        dates_dm = np.array(juld_to_dates(DS_PARAMS['JULD'].values), dtype=object)
        now_dm = datetime.utcnow()
        twelve_mo = timedelta(days=365)

        eligible_arr = np.array([
            (_d is not pd.NaT) and (_d is not None)
            and (now_dm - _d.to_pydatetime() > twelve_mo)
            for _d in dates_dm
        ])
        n_eligible = int(eligible_arr.sum())

        rows_dm = []
        for name, label, *_ in PARAMS:
            modes_dm = get_data_mode_per_param(DS_PARAMS, name)
            n_total_p   = len(modes_dm)
            n_dm_total  = int((modes_dm == 'D').sum())
            n_dm_elig   = int(((modes_dm == 'D') & eligible_arr).sum())
            n_dm_early  = int(((modes_dm == 'D') & ~eligible_arr).sum())
            pct = (n_dm_elig / n_eligible * 100) if n_eligible else 0.0
            rows_dm.append({
                'Parameter':                name,
                'Profiles':                 n_total_p,
                'DMQC done':                n_dm_total,
                'Mature (>12 months old)':  n_eligible,
                'Mature & DMQC done':       n_dm_elig,
                'DMQC coverage':            f'{pct:.1f}%',
                'DMQC done early (<12 months)': n_dm_early,
            })
        st.dataframe(pd.DataFrame(rows_dm),
                     use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Trajectory data
# ═══════════════════════════════════════════════════════════════════════════════
with tab_bgc:
    rtraj_t = ds.get("rtraj")
    dtraj_t = ds.get("dtraj")
    traj_any = dtraj_t if dtraj_t is not None else rtraj_t

    if traj_any is None:
        st.info(
            "**No trajectory file found.**\n\n"
            "This tab requires `Rtraj.nc` (real-time trajectory) "
            "or `Dtraj.nc` (DMQC-processed trajectory). "
            "Core floats may have neither if the data centre has not yet generated them."
        )
    else:
        # ── Top intro ─────────────────────────────────────────────────────────
        st.markdown(
            "Each cycle, the float takes additional measurements beyond the main "
            "upward profile — at the surface, during descent, while drifting at park "
            "depth, and at maximum profile depth. The trajectory file "
            "(`Rtraj.nc` / `Dtraj.nc`) records these scattered measurements along the "
            "float\'s actual path through the ocean."
        )

        # MEASUREMENT_CODE explainer  [#6b/c: renamed, simplified]
        with st.expander("What is a measurement in trajectory file?"):
            st.markdown(
                "Each cycle records measurements at multiple checkpoints:\n\n"
                "| Code | Phase |\n|---|---|\n"
                "| 100  | Descent start (leaves surface) |\n"
                "| 200  | Reached park depth (~1000 dbar) |\n"
                "| 290  | At max depth — dense sampling for sensor calibration ★ |\n"
                "| 300  | Ascent start |\n"
                "| 500  | Near-surface |\n"
                "| 600  | At surface, transmitting |\n"
                "| 703  | First GPS fix after surfacing |\n"
            )
            st.caption(
                "Sometimes at one code, multiple measurements are collected "
                "(especially MC=290 — dense BGC sampling at max depth). "
                "For full code list, see [Argo User\'s Manual Reference Table 15]"
                "(https://vocab.nerc.ac.uk/collection/R15/current/)."
            )

        st.divider()

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  SECTION 1 — Trajectory float track                              ║
        # ║  Left = Rtraj, Right = Dtraj. Color = time, shape = QC           ║
        # ╚══════════════════════════════════════════════════════════════════╝
        st.subheader("Trajectory float track")

        def _extract_track(ds_traj):
            """Return (lat, lon, juld, qc, mc) sorted by time, fill-masked."""
            if ds_traj is None:
                return None
            lat  = ds_traj['LATITUDE'].values.astype(float)
            lon  = ds_traj['LONGITUDE'].values.astype(float)
            juld = ds_traj['JULD'].values.astype(float)
            qc   = decode_bytes(ds_traj['POSITION_QC'].values)
            mc   = ds_traj['MEASUREMENT_CODE'].values

            lat[lat > 9999] = np.nan
            lon[lon > 9999] = np.nan
            juld[juld > 999990] = np.nan

            valid = ~np.isnan(lat) & ~np.isnan(lon) & ~np.isnan(juld)
            lat, lon, juld, qc, mc = lat[valid], lon[valid], juld[valid], qc[valid], mc[valid]

            order = np.argsort(juld)
            return lat[order], lon[order], juld[order], qc[order], mc[order]

        def _date_str(j):
            if np.isnan(j) or j > 999990:
                return '—'
            return (JREF + pd.Timedelta(days=float(j))).strftime('%Y-%m-%d')

        rtrack = _extract_track(rtraj_t)
        dtrack = _extract_track(dtraj_t)

        def _add_track_panel(fig_, track, col_, show_colorbar):
            if track is None:
                return
            lat, lon, juld, qc, mc = track
            if len(lat) == 0:
                return
            n = len(lat)
            time_idx = np.arange(n)
            good = qc == '1'
            bad  = ~good

            def _hover(idx):
                return [
                    f'{_date_str(juld[i])}<br>'
                    f'Lat: {lat[i]:.3f}°  Lon: {lon[i]:.3f}°<br>'
                    f'POSITION_QC: {qc[i]}  MC: {int(mc[i]) if mc[i] < 99999 else "—"}'
                    for i in idx
                ]

            if good.any():
                gi = np.where(good)[0]
                fig_.add_trace(go.Scatter(
                    x=lon[good], y=lat[good], mode='markers',
                    marker=dict(
                        size=6, color=time_idx[good], colorscale='Viridis',
                        cmin=0, cmax=n - 1,
                        showscale=show_colorbar,
                        colorbar=dict(title='Obs index<br>(time →)',
                                      len=0.7, thickness=12,
                                      x=1.02, xanchor='left'),
                        symbol='circle',
                        line=dict(width=0.5, color='white'),
                    ),
                    text=_hover(gi),
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=False,
                ), row=1, col=col_)
            if bad.any():
                bi = np.where(bad)[0]
                fig_.add_trace(go.Scatter(
                    x=lon[bad], y=lat[bad], mode='markers',
                    marker=dict(
                        size=8, color=time_idx[bad], colorscale='Viridis',
                        cmin=0, cmax=n - 1, showscale=False,
                        symbol='diamond',
                        line=dict(width=1, color='black'),
                    ),
                    text=_hover(bi),
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=False,
                ), row=1, col=col_)
            # Start/End triangles
            fig_.add_trace(go.Scatter(
                x=[lon[0]], y=[lat[0]], mode='markers+text',
                marker=dict(size=14, symbol='triangle-up', color='blue'),
                text=['Start'], textposition='top right',
                textfont=dict(size=10),
                showlegend=False, hoverinfo='skip',
            ), row=1, col=col_)
            fig_.add_trace(go.Scatter(
                x=[lon[-1]], y=[lat[-1]], mode='markers+text',
                marker=dict(size=14, symbol='triangle-down', color='red'),
                text=['End'], textposition='bottom right',
                textfont=dict(size=10),
                showlegend=False, hoverinfo='skip',
            ), row=1, col=col_)

        # Build subplot
        if rtrack is None and dtrack is None:
            st.info("Neither Rtraj nor Dtraj provided position fixes.")
        else:
            n_panels = (1 if rtrack is not None else 0) + (1 if dtrack is not None else 0)
            titles = []
            if rtrack is not None:
                titles.append(f'Rtraj  ({len(rtrack[0])} positions)')
            if dtrack is not None:
                titles.append(f'Dtraj  ({len(dtrack[0])} positions)')

            fig_track = make_subplots(
                rows=1, cols=n_panels,
                subplot_titles=titles,
                horizontal_spacing=0.10,
            )
            col_i = 1
            if rtrack is not None:
                _add_track_panel(fig_track, rtrack, col_i,
                                 show_colorbar=(dtrack is None))
                col_i += 1
            if dtrack is not None:
                _add_track_panel(fig_track, dtrack, col_i, show_colorbar=True)

            fig_track.add_annotation(
                xref='paper', yref='paper', x=0.5, y=-0.16,
                text='● circle = good (POSITION_QC=1)   |   ◆ diamond = flagged (POSITION_QC≠1)',
                showarrow=False, font=dict(size=11),
            )
            fig_track.update_xaxes(title_text='Longitude')
            fig_track.update_yaxes(title_text='Latitude', row=1, col=1)
            fig_track.update_layout(
                title=f'Float {WMO} — trajectory positions',
                height=520,
                margin=dict(t=80, b=80, r=120),
            )
            st.plotly_chart(fig_track, use_container_width=True)

            # Bullet description below
            bullet_lines = []
            if rtrack is not None:
                n_r_bad = int(np.sum(rtrack[3] != '1'))
                bullet_lines.append(
                    f"- **Rtraj** (real-time): {len(rtrack[0])} positions; "
                    f"{n_r_bad} with POSITION_QC ≠ 1"
                )
            if dtrack is not None:
                n_d_bad = int(np.sum(dtrack[3] != '1'))
                bullet_lines.append(
                    f"- **Dtraj** (DMQC-processed): {len(dtrack[0])} positions; "
                    f"{n_d_bad} with POSITION_QC ≠ 1"
                )
            st.markdown("\n".join(bullet_lines))

        st.divider()

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  SECTION 2 — Parameters along trajectory                         ║
        # ║  Stacked time series for ALL parameters available in traj file   ║
        # ║  raw=gray, adjusted=black                                        ║
        # ╚══════════════════════════════════════════════════════════════════╝

        # All possible parameters to look for (color now ignored — raw is always gray)
        ALL_TRAJ_PARAMS = [
            ('PRES',  'Pressure',     'dbar'),
            ('TEMP',  'Temperature',  '°C'),
            ('PSAL',  'Salinity',     'PSU'),
            ('DOXY',  'Oxygen',       'µmol/kg'),
            ('NITRATE', 'Nitrate',    'µmol/kg'),
            ('PH_IN_SITU_TOTAL', 'pH', 'total'),
            ('CHLA',  'Chlorophyll',  'mg/m³'),
            ('BBP700', 'Backscatter', 'm⁻¹'),
            ('PPOX_DOXY', 'pO₂',      'mbar'),
        ]
        COLOR_RAW_TRAJ = '#888888'
        COLOR_ADJ_TRAJ = '#000000'

        # Pick whichever traj has the most parameters (prefer Dtraj if both)
        traj_use = traj_any
        traj_label = 'Dtraj' if dtraj_t is not None else 'Rtraj'

        # [#6f] Cache the data extraction so re-rendering this tab is fast
        @st.cache_data(show_spinner=False)
        def _extract_traj_param_data(_ds, base, _wmo):
            """Return dict with raw_dates, raw_vals, adj_dates, adj_vals."""
            juld = _ds['JULD'].values.astype(float)
            juld[juld > 999990] = np.nan
            dates = np.array(
                [JREF + pd.Timedelta(days=float(j)) if not np.isnan(j) else pd.NaT
                 for j in juld], dtype=object,
            )
            valid_date = np.array([d is not pd.NaT and d is not None for d in dates])

            out = {'raw_dates': None, 'raw_vals': None,
                   'adj_dates': None, 'adj_vals': None}
            if base in _ds:
                raw = _ds[base].values.astype(float)
                raw_valid = (raw < 99999) & valid_date
                if raw_valid.any():
                    out['raw_dates'] = dates[raw_valid]
                    out['raw_vals']  = raw[raw_valid]
            adj_var = f'{base}_ADJUSTED'
            if adj_var in _ds:
                adj = _ds[adj_var].values.astype(float)
                adj_valid = (adj < 99999) & valid_date
                if adj_valid.any():
                    out['adj_dates'] = dates[adj_valid]
                    out['adj_vals']  = adj[adj_valid]
            return out

        avail = [(b, l, u) for b, l, u in ALL_TRAJ_PARAMS if b in traj_use]

        # [#6f] Wrap entire section in collapsed expander to keep tab clean
        with st.expander("Parameters along trajectory", expanded=False):
            if not avail:
                st.info(f"No standard parameters found in {traj_label}.")
            else:
                fig_params = make_subplots(
                    rows=len(avail), cols=1,
                    subplot_titles=[f'{label} ({unit})' for _, label, unit in avail],
                    shared_xaxes=True,
                    vertical_spacing=max(0.02, 0.06 / max(len(avail), 1)),
                )

                for row_i, (base, label, unit) in enumerate(avail, start=1):
                    data = _extract_traj_param_data(traj_use, base, WMO)

                    # Raw markers (gray)
                    if data['raw_vals'] is not None:
                        fig_params.add_trace(go.Scatter(
                            x=data['raw_dates'], y=data['raw_vals'],
                            mode='markers',
                            marker=dict(size=2, color=COLOR_RAW_TRAJ, opacity=0.5),
                            name='raw',
                            showlegend=(row_i == 1),
                            legendgroup='raw',
                            hovertemplate=(f'{label} raw: %{{y:.3f}} {unit}<br>'
                                           'Date: %{x|%Y-%m-%d}<extra></extra>'),
                        ), row=row_i, col=1)

                    # Adjusted markers (black)
                    if data['adj_vals'] is not None:
                        fig_params.add_trace(go.Scatter(
                            x=data['adj_dates'], y=data['adj_vals'],
                            mode='markers',
                            marker=dict(size=2.5, color=COLOR_ADJ_TRAJ, opacity=0.85),
                            name='adjusted',
                            showlegend=(row_i == 1),
                            legendgroup='adj',
                            hovertemplate=(f'{label} adj: %{{y:.3f}} {unit}<br>'
                                           'Date: %{x|%Y-%m-%d}<extra></extra>'),
                        ), row=row_i, col=1)

                    fig_params.update_yaxes(title_text=unit, row=row_i, col=1)

                fig_params.update_xaxes(title_text='Date', row=len(avail), col=1)
                fig_params.update_layout(
                    title=(f'Float {WMO} — parameters along trajectory  ({traj_label})  '
                           '<span style="font-size:11px;color:#666">'
                           'gray = raw   |   black = adjusted (DMQC)</span>'),
                    height=max(280, 180 * len(avail)),
                    showlegend=True,
                    legend=dict(orientation='h', y=-0.02),
                    margin=dict(t=70, b=80),
                )
                st.plotly_chart(fig_params, use_container_width=True)

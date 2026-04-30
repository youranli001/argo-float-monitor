# Argo Float Monitoring Dashboard

An interactive Streamlit dashboard for monitoring any Argo profiling float — metadata,
engineering health, data delivery, profiles, QC, and trajectory measurements.
Built as a demonstration of PMEL-style fleet monitoring capability.

**Live demo:** [argo-float-monitor.streamlit.app](https://argo-float-monitor.streamlit.app)

## Usage

### Option 1 — Use the live app

Open the live demo, enter any WMO number, and click Download.
The app fetches all standard files directly from the IFREMER GDAC FTP server.

### Option 2 — Run locally

```bash
git clone https://github.com/youranli001/argo-float-monitor.git
cd argo-float-monitor
pip install -r requirements.txt
streamlit run argo_monitor.py
```

Running locally lets you point at NetCDF files already on disk, or download any float
by WMO number from GDAC.

## Dashboard tabs

**Main Information**
- Header
- Interactive global map of profile trajectories
- Main info, tracking lifecycle, deployment, and cycle activity cards

**Float Metadata** (`meta.nc`)
- Argo project information
- Platform information
- Deployment information
- Sensors
- Factory calibration
- Launch & mission configuration parameters

**Float Health** (`tech.nc`)
- Health summary with customized diagnoses
- Pressure (surface offset, internal vacuum, air bladder)
- Buoyancy (pump activity with linear trend)
- Battery (voltage and current under different loads)
- Communication & timing
- Repositions & drift
- Ice detection
- Piston positions (Now / Surface / Park)
- Decoded status flags

**Profiles**
- Section plots: depth–time heatmap per parameter
- All-profiles overlay grid: every cycle drawn together
- Single-cycle profile: T / S / T-S diagram + BGC parameters, with per-parameter DATA_MODE annotation

**QC**
- Data mode of all cycles, grouped by parameter family
- Per-parameter QC heatmaps (depth × cycle) with vertical legends for QC flag, Profile QC, and DATA_MODE
- Scientific calibration records (per-cycle equation / coefficient / comment)
- Subsurface velocity quality (reposition counting from `tech.nc`)

**Data Delivery**
- Variables and parameters table with P02 vocabulary references
- Surface-to-GDAC delivery latency benchmarked against the 12-hour Argo target
- Delayed-mode eligibility coverage table

**Trajectory data** (`Rtraj.nc` / `Dtraj.nc`)
- MEASUREMENT_CODE reference (descent, park, deep stop, surface, GPS fix)
- Float track map with Rtraj vs Dtraj comparison and POSITION_QC markers
- Parameters along trajectory: stacked time series, raw vs DMQC-adjusted

## Author

Youran Li · [github.com/youranli001](https://github.com/youranli001)

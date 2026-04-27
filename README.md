# Argo Float Monitoring Dashboard

An interactive Streamlit dashboard for monitoring any Argo profiling float — health, data delivery, and BGC profiles. Built as a demonstration of PMEL-style fleet monitoring capability.

## Installation

```bash
pip install -r requirements.txt
streamlit run argo_monitor.py
```

## Data access

The sidebar offers two modes:

- **Local files** — point to a folder containing the float NetCDF files (e.g. `5906551_prof.nc`, `5906551_tech.nc`)
- **Download from GDAC** — enter any WMO number and click Download; the app connects to the IFREMER FTP server (`ftp.ifremer.fr`), locates the float across all DACs, and downloads all standard files to a local cache (`~/.argo_cache/{wmo}/`)

Any float in the Argo network can be loaded by WMO number.

## Dashboard tabs

**Float Track** — profile positions on an interactive map, colored by cycle number, with start/end markers and position summary statistics

**Float Health** — engineering telemetry from `_tech.nc`:
- Buoyancy pump on-time with linear trend (increasing trend = early degradation signal)
- Battery voltage and current
- Internal vacuum (sudden drop = water intrusion risk)
- Surface pressure offset (flag threshold: ±20 dbar)
- Reposition count per cycle (>0 = subsurface velocity estimate unreliable)
- CTD status hex flag anomalies decoded and tabulated

**Data Delivery** — per-cycle transmission delay from ascent end to first Iridium transmission, compared against the 12-hour Argo real-time target; includes delay distribution histogram and subsurface velocity QC summary

**Profile Explorer** — for any selected cycle:
- Temperature and salinity depth profiles (raw vs DMQC-adjusted)
- T-S diagram (water mass fingerprint)
- BGC parameters (O₂, Chl-a, NO₃⁻, pH, BBP700) from `_Sprof.nc` if available
- SCIENTIFIC_CALIB records expandable per cycle
- DATA_MODE timeline across all cycles
- All-cycles T/S overlay (colored blue→red, early→late)

**BGC Time Series** — BGC parameters along trajectory from `_Dtraj.nc`, showing raw vs adjusted values over the float lifetime; plus BGC depth profiles for any selected cycle

## Author

Youran Li · [github.com/youranli001](https://github.com/youranli001)
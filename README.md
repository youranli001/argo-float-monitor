# Argo Float Monitoring Dashboard

Interactive dashboard for monitoring Argo float health and data delivery,
built as a demonstration of PMEL-style fleet monitoring capability.

**Float 5906551** — GO-BGC float, Southern Ocean (~40°S)  
Sensors: T / S / P · O₂ · Chl-a · BBP700 · pH · NO₃⁻  
117 cycles · 2021–2026 · Data: US GDAC / AOML

---

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
streamlit run argo_monitor.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

Set the **data directory** in the sidebar to the folder containing your
float NetCDF files (e.g. `5906551_prof.nc`, `5906551_tech.nc`, etc.).

---

## Dashboard Panels

### 📍 Float Track
Map of profile locations colored by cycle number, with start/end markers
and position summary statistics.

### 🔧 Float Health
Engineering telemetry from `_tech.nc`:

| Panel | What it shows |
|---|---|
| Buoyancy pump on-time | **Increasing trend = early buoyancy degradation signal.** This float shows +2.4 s/cycle — visible before battery or pressure data show anomalies. |
| Battery voltage | Stepwise decline = normal battery pack depletion |
| Battery current | Declining trend = aging |
| Internal vacuum | Linear decline = normal; sudden drop = water intrusion |
| Pressure offset | Should stay near 0 dbar; >±20 dbar = investigate |
| Reposition count | >0 = subsurface velocity estimate unreliable for that cycle |

CTD status hex flags are decoded and tabulated when non-zero.

### 📡 Data Delivery
Per-cycle transmission delay (ascent end → first Iridium transmission),
compared against the 12-hour Argo target. Includes:
- Timeline bar chart (green = on-time, red = late)
- Delay distribution histogram
- Subsurface velocity QC analysis (reposition contamination)

### 🌡️ Profile Explorer
Interactive T / S / T-S diagram for any selected cycle, with:
- Raw vs DMQC-adjusted variables overlaid
- BGC parameters (O₂, Chl-a, NO₃⁻, pH) if Sprof.nc is available
- Calibration records (SCIENTIFIC_CALIB) expandable per cycle
- DATA_MODE timeline for all cycles

---

## Key Findings — Float 5906551

- **Pump time trend: +2.4 s/cycle** → early buoyancy degradation signal
- Pressure offset stable near 0 dbar — CTD pressure sensor healthy
- 2 cycles with non-zero CTD status flags — one graded B, one A (inconsistency)
- CTD status bit meaning not publicly documented (requires APEX firmware docs)
- Reposition count > 0 on several cycles → velocity estimates contaminated;
  no dedicated velocity QC flag in Argo format (gap)

---

## Author

Youran Li · [github.com/youranli001](https://github.com/youranli001)

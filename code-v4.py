# ============================================================
# REPEATED DROUGHT AND ECOSYSTEM RECOVERY
# Corrected primary analysis
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3
})


# ============================================================
# 1. CONFIGURATION
# ============================================================

CFG = dict(
    # ========================================================
    # Drought definition
    # ========================================================
    spei_threshold = -1.0,
    merge_gap      = 2,
    min_duration   = 2,

    # ========================================================
    # Vegetation response
    # ========================================================
    lag = 1,

    # ========================================================
    # Pre/post windows
    # ========================================================
    pre_window  = 6,
    post_window = 6,

    # ========================================================
    # Recovery definition
    # ========================================================
    recovery_ratio = 0.90,
    confirm_months = 2,
    max_recovery_months = 24,

    # ========================================================
    # Common analysis period
    # ========================================================
    analysis_start = "2001-01-01",
    analysis_end   = "2022-12-01",

    # ========================================================
    # Primary analysis
    # ========================================================
    detrend = False,

    # ========================================================
    # Bootstrap
    # ========================================================
    n_boot = 5000,
    random_seed = 42,

    # ========================================================
    # Alternative repeat definition
    # Sensitivity analysis ONLY
    # ========================================================
    alternative_repeat_gap = 24
)


DATA_DIR = os.environ.get("DATA_DIR", ".")


# ============================================================
# 2. FILES
# ============================================================

DISTRICTS = {

    "Anantapur": {
        "ndvi": "Anantapur_MODIS_NDVI_2000_2026.csv",
        "gpp": "Anantapur_MODIS_GPP_2001_2026_STITCHED.csv",
        "rain": "Anantapur_CHIRPS_Rainfall_2000_2026.csv",
        "spei": "Anantapur_SPEI_03_2000_2026.csv",
        "era5": "Anantapur_ERA5Land_2001_2026.csv"
    },

    "Kurnool": {
        "ndvi": "Kurnool_MODIS_NDVI_2001_2026.csv",
        "gpp": "Kurnool_MODIS_GPP_2001_2026_STITCHED.csv",
        "rain": "Kurnool_CHIRPS_Rainfall_2001_2026.csv",
        "spei": "Kurnool_SPEI_03_2001_2023.csv",
        "era5": "Kurnool_ERA5Land_2001_2026.csv"
    },

    "East Godavari": {
        "ndvi": "EastGodavari_MODIS_NDVI_2001_2026.csv",
        "gpp": "EastGodavari_MODIS_GPP_2001_2026_STITCHED.csv",
        "rain": "EastGodavari_CHIRPS_Rainfall_2001_2026.csv",
        "spei": "EastGodavari_SPEI_03_2001_2023.csv",
        "era5": "EastGodavari_ERA5Land_2001_2026.csv"
    },
    
    "Kadapa": {
        "ndvi": "Kadapa_MODIS_NDVI_2001_2026.csv",
        "gpp": "Kadapa_MODIS_GPP_2001_2026_STITCHED.csv",
        "rain": "Kadapa_CHIRPS_Rainfall_2001_2026.csv",
        "spei": "Kadapa_SPEI_03_2001_2023.csv",
        "era5": "Kadapa_ERA5Land_2001_2026.csv"          
    },
    
    "Chittoor": {
        "ndvi": "Chittoor_MODIS_NDVI_2001_2025_MONTHLY.csv",
        "gpp": "Chittoor_MODIS_GPP_2001_2025_MONTHLY.csv",
        "rain": "Chittoor_CHIRPS_RAINFALL_2001_2025_MONTHLY.csv",
        "spei": "Chittoor_SPEI_2001_2025_MONTHLY.csv",
        "era5": "Chittoor_ERA5_LAND_2001_2025_MONTHLY.csv"          
    },
    
    "Prakasam": {
        "ndvi": "Prakasam_MODIS_NDVI_2001_2025_MONTHLY.csv",
        "gpp": "Prakasam_MODIS_GPP_2001_2025_MONTHLY.csv",
        "rain": "Prakasam_CHIRPS_RAINFALL_2001_2025_MONTHLY.csv",
        "spei": "Prakasam_SPEI_2001_2025_MONTHLY.csv",
        "era5": "Prakasam_ERA5_LAND_2001_2025_MONTHLY.csv"          
    },
    
}


# ============================================================
# 3. FILE LOADING
# ============================================================

def find_file(fname):

    for base in (DATA_DIR, "."):

        path = os.path.join(base, fname)

        if os.path.exists(path):
            return path

    # Kaggle compatibility
    if os.path.exists("/kaggle/input"):

        for root, _, files in os.walk("/kaggle/input"):

            if fname in files:
                return os.path.join(root, fname)

    raise FileNotFoundError(
        f"{fname} not found. "
        f"Set DATA_DIR or upload the file."
    )


def read_col(fname, col, new_name):

    df = pd.read_csv(find_file(fname))

    if "date" not in df.columns:
        raise KeyError(
            f"'date' column missing from {fname}. "
            f"Columns: {list(df.columns)}"
        )

    if col not in df.columns:
        raise KeyError(
            f"'{col}' not found in {fname}. "
            f"Columns: {list(df.columns)}"
        )

    s = df[["date", col]].copy()

    s["date"] = pd.to_datetime(
        s["date"],
        errors="coerce"
    )

    s[col] = pd.to_numeric(
        s[col],
        errors="coerce"
    )

    s = s.dropna(subset=["date"])

    return (
        s.set_index("date")[col]
        .rename(new_name)
        .sort_index()
    )

def build_district(files):

    # ---------- NDVI ----------
    ndvi = read_col(
        files["ndvi"],
        "ndvi_mean",
        "ndvi"
    )

    # ---------- GPP ----------
    gpp = read_col(
        files["gpp"],
        "gpp_mean_kg_C_m2",
        "gpp"
    )

    # ---------- CHIRPS rainfall ----------
    rain_file = pd.read_csv(files["rain"], nrows=2)

    if "rainfall_mm_month" in rain_file.columns:
        rain_col = "rainfall_mm_month"
    elif "precipitation_mm" in rain_file.columns:
        rain_col = "precipitation_mm"
    else:
        raise KeyError(
            f"No rainfall column found in {files['rain']}. "
            f"Columns: {list(rain_file.columns)}"
        )

    rain = read_col(
        files["rain"],
        rain_col,
        "rain"
    )

    # ---------- SPEI ----------
    spei_file = pd.read_csv(files["spei"], nrows=2)

    if "spei" in spei_file.columns:
        spei_col = "spei"
    elif "spei_03" in spei_file.columns:
        spei_col = "spei_03"
    else:
        raise KeyError(
            f"No SPEI column found in {files['spei']}. "
            f"Columns: {list(spei_file.columns)}"
        )

    spei = read_col(
        files["spei"],
        spei_col,
        "spei"
    )

    # ---------- ERA5 ----------
    era5_file = pd.read_csv(files["era5"], nrows=2)

    if "era5_temp_C" in era5_file.columns:
        # New GEE files: already Celsius
        temp = read_col(
            files["era5"],
            "era5_temp_C",
            "temp"
        )

    elif "temperature_2m_K" in era5_file.columns:
        # Old files: Kelvin -> Celsius
        temp = read_col(
            files["era5"],
            "temperature_2m_K",
            "temp"
        )
        temp = temp - 273.15

    else:
        raise KeyError(
            f"No temperature column found in {files['era5']}. "
            f"Columns: {list(era5_file.columns)}"
        )

    # ---------- Combine ----------
    series = [ndvi, gpp, rain, spei, temp]

    m = pd.concat(
        [s.resample("MS").mean() for s in series],
        axis=1
    ).asfreq("MS")

    # Fill only short internal gaps
    m = m.interpolate(
        method="time",
        limit=2,
        limit_area="inside"
    )

    # Keep only period where SPEI and GPP are available
    ok = m[["spei", "gpp"]].dropna().index

    if len(ok) == 0:
        raise ValueError(
            "No overlapping SPEI/GPP period found."
        )

    return m.loc[ok.min():ok.max()]

MONTHLY = {
    name: build_district(files)
    for name, files in DISTRICTS.items()
}

# ============================================================
# COMMON ANALYSIS PERIOD
# ============================================================

START_DATE = pd.Timestamp(CFG["analysis_start"])
END_DATE   = pd.Timestamp(CFG["analysis_end"])

for name in MONTHLY:

    MONTHLY[name] = MONTHLY[name].loc[
        START_DATE:END_DATE
    ].copy()

print("\n================ COMMON PERIOD ================\n")

for name, m in MONTHLY.items():

    print(
        f"{name:15s}"
        f"{m.index.min().date()} -> "
        f"{m.index.max().date()} | "
        f"{len(m)} months"
    )

print("\n================ DATA COVERAGE ================\n")

for name, m in MONTHLY.items():

    print(
        f"{name:15s}"
        f"{m.index.min().date()} -> "
        f"{m.index.max().date()} | "
        f"{len(m)} months | "
        f"missing = {m.isna().sum().to_dict()}"
    )


def assign_repeat_status(events):

    events = events.copy()

    events["repeat"] = 0

    for i in range(1, len(events)):

        previous_recovery = (
            events.loc[
                i - 1,
                "recovery_date"
            ]
        )

        current_start = (
            events.loc[
                i,
                "start"
            ]
        )

        # ----------------------------------------------------
        # Main definition:
        # Current drought begins before recovery
        # from previous drought is complete.
        # ----------------------------------------------------

        if pd.notna(previous_recovery):

            if current_start <= previous_recovery:

                events.loc[
                    i,
                    "repeat"
                ] = 1

    return events

def add_alternative_repeat_flag(events, cfg):

    events = events.copy()

    events["repeat_24m"] = 0

    for i in range(1, len(events)):

        gap = events.loc[
            i,
            "gap_months"
        ]

        if pd.notna(gap):

            if (
                gap <=
                cfg["alternative_repeat_gap"]
            ):

                events.loc[
                    i,
                    "repeat_24m"
                ] = 1

    return events

def bootstrap_difference(
    repeat_values,
    isolated_values,
    n_boot=5000,
    seed=42
):

    repeat_values = np.asarray(
        repeat_values,
        dtype=float
    )

    isolated_values = np.asarray(
        isolated_values,
        dtype=float
    )

    repeat_values = repeat_values[
        np.isfinite(repeat_values)
    ]

    isolated_values = isolated_values[
        np.isfinite(isolated_values)
    ]

    if (
        len(repeat_values) < 2
        or len(isolated_values) < 2
    ):
        return {
            "difference": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "n_repeat": len(repeat_values),
            "n_isolated": len(isolated_values)
        }

    rng = np.random.default_rng(seed)

    observed = (
        repeat_values.mean()
        - isolated_values.mean()
    )

    boot = np.empty(n_boot)

    for b in range(n_boot):

        r = rng.choice(
            repeat_values,
            size=len(repeat_values),
            replace=True
        )

        s = rng.choice(
            isolated_values,
            size=len(isolated_values),
            replace=True
        )

        boot[b] = (
            r.mean()
            - s.mean()
        )

    return {
        "difference": observed,
        "ci_low": np.percentile(
            boot,
            2.5
        ),
        "ci_high": np.percentile(
            boot,
            97.5
        ),
        "n_repeat": len(repeat_values),
        "n_isolated": len(isolated_values)
    }
    
    
def find_recovery(
    rel_series,
    event_end,
    cfg
):
    """
    Find first recovery point after drought.

    Recovery:
        GPP >= recovery_ratio
        for confirm_months consecutive months.

    Returns:
        recovery_date
        recovery_time_months
        loss_detected
    """

    ratio = rel_series.dropna()

    # --------------------------------------------------------
    # Response window
    # --------------------------------------------------------

    response_start = (
        event_end
        + pd.DateOffset(months=cfg["lag"])
    )

    response_end = (
        response_start
        + pd.DateOffset(
            months=cfg["post_window"] - 1
        )
    )

    response = ratio.loc[
        response_start:response_end
    ]

    if response.empty:
        return pd.NaT, np.nan, False

    # --------------------------------------------------------
    # Detect meaningful loss
    # --------------------------------------------------------

    loss_detected = (
        response.min()
        < cfg["recovery_ratio"]
    )

    # No loss -> recovery time is not meaningful
    if not loss_detected:

        return (
            pd.NaT,
            np.nan,
            False
        )

    # --------------------------------------------------------
    # Search for recovery
    # --------------------------------------------------------

    search_end = (
        event_end
        + pd.DateOffset(
            months=cfg["max_recovery_months"]
        )
    )

    recovery_series = ratio.loc[
        response_start:search_end
    ]

    dates = recovery_series.index

    required = cfg["confirm_months"]

    for i in range(
        len(recovery_series) - required + 1
    ):

        window = recovery_series.iloc[
            i:i + required
        ]

        if (
            window >= cfg["recovery_ratio"]
        ).all():

            recovery_date = dates[i]

            recovery_months = (
                recovery_date.year
                - event_end.year
            ) * 12 + (
                recovery_date.month
                - event_end.month
            )

            return (
                recovery_date,
                recovery_months,
                True
            )

    # No recovery within 24 months
    return (
        pd.NaT,
        np.nan,
        True
    )

# ============================================================
# 4. SEASONS
# ============================================================

SEASON = {

    12: "Winter",
    1: "Winter",
    2: "Winter",

    3: "Pre-monsoon",
    4: "Pre-monsoon",
    5: "Pre-monsoon",

    6: "Monsoon",
    7: "Monsoon",
    8: "Monsoon",
    9: "Monsoon",

    10: "Post-monsoon",
    11: "Post-monsoon"
}


# ============================================================
# 5. DROUGHT EVENT DETECTION
# ============================================================

def detect_events(m, cfg):

    drought_month = (
        m["spei"] < cfg["spei_threshold"]
    ).fillna(False).values

    # --------------------------------------------------------
    # Find continuous drought runs
    # --------------------------------------------------------

    runs = []

    i = 0

    while i < len(drought_month):

        if drought_month[i]:

            j = i

            while (
                j + 1 < len(drought_month)
                and drought_month[j + 1]
            ):
                j += 1

            runs.append([i, j])

            i = j + 1

        else:
            i += 1

    # --------------------------------------------------------
    # Merge nearby drought runs
    # --------------------------------------------------------

    merged = []

    for start_i, end_i in runs:

        if (
            merged
            and
            start_i - merged[-1][1] - 1
            <= cfg["merge_gap"]
        ):
            merged[-1][1] = end_i

        else:
            merged.append([start_i, end_i])

    # --------------------------------------------------------
    # Create event table
    # --------------------------------------------------------

    rows = []

    for start_i, end_i in merged:

        duration = end_i - start_i + 1

        if duration < cfg["min_duration"]:
            continue

        spei = m["spei"].iloc[
            start_i:end_i + 1
        ]

        rows.append({
            "start_i": start_i,
            "end_i": end_i,

            "start": m.index[start_i],
            "end": m.index[end_i],

            "duration": duration,

            "min_spei": float(spei.min()),

            "severity": float(
                (-spei.clip(upper=0)).sum()
            )
        })

    ev = pd.DataFrame(rows)

    if ev.empty:
        return ev

    ev["order"] = np.arange(
        1,
        len(ev) + 1
    )

    # --------------------------------------------------------
    # Correct meaning of gap_months
    #
    # Number of months between previous drought END
    # and current drought START.
    # --------------------------------------------------------

    ev["gap_months"] = (
        ev["start_i"]
        - ev["end_i"].shift(1)
        - 1
    )

    ev["prior_severity"] = (
        ev["severity"].shift(1)
    )

    # Repeat is assigned AFTER recovery is calculated.
    ev["repeat"] = 0

    return ev

# ============================================================
# 6. CREATE SEASONAL GPP / NDVI BASELINES
# ============================================================

def add_baselines(m, events):

    o = m.copy()

    # Mark drought months
    o["drought"] = False

    for _, e in events.iterrows():

        o.loc[
            e["start"]:e["end"],
            "drought"
        ] = True


    # --------------------------------------------------------
    # Seasonal baseline
    #
    # IMPORTANT:
    # Baseline is calculated from NON-DROUGHT months only.
    # --------------------------------------------------------

    for variable in ["gpp", "ndvi", "soil", "temp"]:

        if variable not in o.columns:
            continue

        valid = o.loc[
            ~o["drought"],
            variable
        ]

        baseline_by_month = (
            valid
            .groupby(valid.index.month)
            .median()
        )

        o[
            variable + "_baseline"
        ] = o.index.month.map(
            baseline_by_month
        )

        o[
            variable + "_rel"
        ] = (
            o[variable]
            /
            o[variable + "_baseline"]
        )


        # Seasonal z-score
        month_mean = (
            valid
            .groupby(valid.index.month)
            .transform("mean")
        )

        month_sd = (
            valid
            .groupby(valid.index.month)
            .transform("std")
        )

        # Map the monthly statistics back to the full index
        mean_map = {
            month: value
            for month, value
            in valid.groupby(
                valid.index.month
            ).mean().items()
        }

        sd_map = {
            month: value
            for month, value
            in valid.groupby(
                valid.index.month
            ).std().items()
        }

        o[
            variable + "_z"
        ] = (
            o[variable]
            -
            o.index.month.map(mean_map)
        ) / o.index.month.map(sd_map)


    return o


# ============================================================
# 7. OPTIONAL LINEAR DETRENDING
# ============================================================

def detrend_series(o, variable):

    rel_col = variable + "_rel"

    if rel_col not in o.columns:
        return

    y = o[rel_col].values

    ok = np.isfinite(y)

    if ok.sum() < 3:
        return

    t = np.arange(len(o))

    coef = np.polyfit(
        t[ok],
        y[ok],
        1
    )

    trend = np.polyval(
        coef,
        t
    )

    # Keep mean approximately unchanged
    o[rel_col] = (
        o[rel_col]
        -
        (trend - np.nanmean(o[rel_col]))
    )


# ============================================================
# 8. RECOVERY DETECTION
# ============================================================

def find_recovery(
    ratio,
    start_index,
    end_index,
    max_months,
    threshold,
    confirm
):

    n = len(ratio)

    search_start = end_index + 1

    search_end = min(
        end_index + max_months,
        n - 1
    )

    if search_start > search_end:
        return None

    consecutive = 0

    for j in range(
        search_start,
        search_end + 1
    ):

        value = ratio.iloc[j]

        if pd.notna(value) and value >= threshold:

            consecutive += 1

            if consecutive >= confirm:

                # First month of the confirmed recovery
                return j - confirm + 1

        else:

            consecutive = 0

    return None


# ============================================================
# 9. COMPUTE EVENT METRICS
# ============================================================

def compute_event_metrics(
    m,
    events,
    cfg
):

    rows = []

    n = len(m)

    gpp_ratio = m["gpp_rel"]

    ndvi_ratio = m["ndvi_rel"]

    for k, event in events.iterrows():

        s = int(event["start_i"])
        e = int(event["end_i"])

        # Vegetation-response window
        response_end = min(
            e + cfg["lag"],
            n - 1
        )

        row = event.to_dict()

        row["season"] = SEASON[
            event["start"].month
        ]

        row["onset_month"] = (
            event["start"].month
        )


        # ----------------------------------------------------
        # RESISTANCE
        #
        # Minimum GPP / seasonal baseline
        # ----------------------------------------------------

        gpp_window = gpp_ratio.iloc[
            s:response_end + 1
        ]

        row["resist"] = (
            gpp_window.min()
        )


        # NDVI resistance for validation
        if ndvi_ratio is not None:

            ndvi_window = ndvi_ratio.iloc[
                s:response_end + 1
            ]

            row["ndvi_resist"] = (
                ndvi_window.min()
            )


        # ----------------------------------------------------
        # PRE-DROUGHT CONDITION
        # ----------------------------------------------------

        pre_start = max(
            0,
            s - cfg["pre_window"]
        )

        pre = m["gpp_z"].iloc[
            pre_start:s
        ]

        row["pre_z"] = (
            pre.mean()
            if len(pre)
            else np.nan
        )


        # ----------------------------------------------------
        # CLIMATE CONDITIONS DURING EVENT
        # ----------------------------------------------------

        # row["soil_z_mean"] = (
        #     m["soil_z"].iloc[
        #         s:response_end + 1
        #     ].mean()
        # )

        row["temp_z_mean"] = (
            m["temp_z"].iloc[
                s:response_end + 1
            ].mean()
        )


        # ----------------------------------------------------
        # GPP LOSS
        #
        # Actual ecological threshold:
        # resistance < 0.90
        # ----------------------------------------------------

        row["had_loss"] = int(
            pd.notna(row["resist"])
            and
            row["resist"] < cfg["recovery_ratio"]
        )


        # ----------------------------------------------------
        # RECOVERY WITHIN 24 MONTHS
        #
        # This is calculated independently of the next drought.
        # ----------------------------------------------------

        if row["had_loss"] == 1:

            recovery_index = find_recovery(
                gpp_ratio,
                s,
                response_end,
                cfg["max_recovery_months"],
                cfg["recovery_ratio"],
                cfg["confirm_months"]
            )

            if recovery_index is not None:

                row["recovery_date"] = (
                    m.index[recovery_index]
                )

                row["rec_time"] = (
                    recovery_index
                    - response_end
                )

                row["recovered"] = 1

            else:

                row["recovery_date"] = pd.NaT

                row["rec_time"] = np.nan

                row["recovered"] = 0

        else:

            # No GPP loss occurred, so recovery time is not applicable
            row["recovery_date"] = pd.NaT
            row["rec_time"] = np.nan
            row["recovered"] = np.nan


        # ----------------------------------------------------
        # POST-DROUGHT RESPONSE
        # ----------------------------------------------------

        post_start = (
            response_end + 1
        )

        post_end = min(
            response_end
            + cfg["post_window"],
            n - 1
        )

        if post_start <= post_end:

            post = m["gpp_z"].iloc[
                post_start:post_end + 1
            ]

            row["post_z"] = post.mean()

        else:

            row["post_z"] = np.nan


        # ----------------------------------------------------
        # RECOVERY GAIN
        # ----------------------------------------------------

        pre_recovery_z = (
            m["gpp_z"].iloc[
                s:response_end + 1
            ].mean()
        )

        if pd.notna(row["post_z"]):

            row["recovery_gain"] = (
                row["post_z"]
                -
                pre_recovery_z
            )

        else:

            row["recovery_gain"] = np.nan


        # ----------------------------------------------------
        # GAP TO NEXT DROUGHT
        # ----------------------------------------------------

        if k + 1 < len(events):

            next_start = int(
                events["start_i"].iloc[k + 1]
            )

            row["next_start_i"] = next_start

            row["next_start"] = (
                events["start"].iloc[k + 1]
            )

            row["gap_months"] = (
                next_start
                -
                e
            )

        else:

            next_start = None

            row["next_start_i"] = np.nan

            row["next_start"] = pd.NaT

            row["gap_months"] = np.nan


        # ----------------------------------------------------
        # CRITICAL:
        #
        # REPEAT = current drought occurs BEFORE the
        # previous drought has recovered.
        #
        # This is NOT based on a 24-month gap.
        # ----------------------------------------------------

        if k == 0:

            # First drought has no previous drought
            row["repeat"] = 0

        else:

            previous_event = rows[-1]

            previous_had_loss = previous_event["had_loss"]

            previous_recovery_date = previous_event[
                "recovery_date"
            ]

            current_start = event["start"]

            # ----------------------------------------------------
            # REPEAT DEFINITION
            #
            # A drought is repeated only when it begins before
            # the ecosystem has recovered from a previous drought
            # that actually caused a GPP loss.
            # ----------------------------------------------------

            if previous_had_loss == 0:

                # Previous drought caused no GPP loss.
                # Therefore there is no recovery period to wait for.
                row["repeat"] = 0

            elif pd.isna(previous_recovery_date):

                # Previous drought caused a GPP loss but did not
                # recover within the 24-month observation window.
                row["repeat"] = 1

            else:

                # Previous drought recovered.
                # Current drought is repeated only if it begins
                # before that recovery date.
                row["repeat"] = int(
                    current_start
                    <=
                    previous_recovery_date
                )


        rows.append(row)


    return pd.DataFrame(rows)

def clustered_regression(
    df,
    formula,
    cluster_column="year"
):

    d = df.copy()

    d["year"] = (
        pd.to_datetime(d["start"])
        .dt.year
    )

    d = d.dropna(
        subset=[
            "resist",
            "severity",
            "duration",
            "repeat",
            "district",
            "year"
        ]
    )

    if d["year"].nunique() < 5:

        print(
            "Too few year clusters for "
            "clustered regression."
        )

        return None

    model = smf.ols(
        formula,
        data=d
    ).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": d["year"]
        }
    )

    return model

# ============================================================
# 10. RUN ALL DISTRICTS
# ============================================================

all_metrics = []
ANOM = {}

for district, raw in MONTHLY.items():

    # Detect drought events first
    events = detect_events(
        raw,
        CFG
    )

    if events.empty:
        continue

    # Build seasonal baseline
    m = add_baselines(
        raw,
        events
    )

    # Primary analysis does NOT detrend
    if CFG["detrend"]:

        detrend_series(
            m,
            "gpp"
        )

        detrend_series(
            m,
            "ndvi"
        )

    # Compute event metrics
    metrics = compute_event_metrics(
        m,
        events,
        CFG
    )

    metrics.insert(
        0,
        "district",
        district
    )

    all_metrics.append(
        metrics
    )

    ANOM[district] = (
        m,
        metrics
    )


MET = pd.concat(
    all_metrics,
    ignore_index=True
)


# ============================================================
# 11. MAIN RESULTS
# ============================================================

print("\n================ MAIN RESULTS ================\n")

print(
    f"Total drought events: {len(MET)}"
)

print("\nEvents by district:")

print(
    MET.groupby("district").agg(
        events=("order", "size"),
        repeat=("repeat", "sum"),
        losses=("had_loss", "sum")
    )
)


print(
    "\nEvents with GPP below 90% of baseline:"
)

print(
    f"{int(MET['had_loss'].sum())} "
    f"of {len(MET)}"
)


print(
    "\nUnrecovered/censored events "
    f"(no recovery within {CFG['max_recovery_months']} months):"
)

print(
    int(
        (MET["recovered"] == 0).sum()
    )
)


# ============================================================
# 12. EVENT TABLE
# ============================================================

cols = [

    "district",
    "order",
    "start",
    "end",

    "duration",
    "min_spei",
    "severity",

    "gap_months",
    "repeat",
    "season",

    "resist",
    "had_loss",

    "recovery_date",
    "rec_time",
    "recovered",

    "pre_z",
    "post_z",
    "recovery_gain",

    # "soil_z_mean",
    "temp_z_mean"
]


print(
    "\n================ EVENT TABLE ================\n"
)

print(
    MET[
        [c for c in cols if c in MET.columns]
    ].round(3).to_string(index=False)
)


# Save
MET.to_csv(
    "drought_event_metrics_corrected.csv",
    index=False
)


# ============================================================
# 13. REPEAT-EVENT COUNTS
# ============================================================

print(
    "\n================ REPEAT CLASSIFICATION ================\n"
)

print(
    MET[
        [
            "district",
            "order",
            "start",
            "repeat",
            "recovery_date",
            "rec_time"
        ]
    ].to_string(index=False)
)


print("\nRepeat counts:")

print(
    MET["repeat"].value_counts()
)


# ============================================================
# 14. SIMPLE GROUP COMPARISON
# ============================================================

def group_summary(df, variable):

    repeat_values = (
        df.loc[
            df["repeat"] == 1,
            variable
        ]
        .dropna()
        .values
    )

    isolated_values = (
        df.loc[
            df["repeat"] == 0,
            variable
        ]
        .dropna()
        .values
    )

    result = {
        "variable": variable,
        "n_repeat": len(repeat_values),
        "n_isolated": len(isolated_values),
        "mean_repeat": np.nan,
        "mean_isolated": np.nan,
        "median_repeat": np.nan,
        "median_isolated": np.nan,
        "difference": np.nan,
        "p_mw": np.nan,
        "cliffs_delta": np.nan
    }

    if len(repeat_values):

        result["mean_repeat"] = (
            np.mean(repeat_values)
        )

        result["median_repeat"] = (
            np.median(repeat_values)
        )

    if len(isolated_values):

        result["mean_isolated"] = (
            np.mean(isolated_values)
        )

        result["median_isolated"] = (
            np.median(isolated_values)
        )

    if (
        len(repeat_values) >= 2
        and
        len(isolated_values) >= 2
    ):

        result["difference"] = (
            np.mean(repeat_values)
            -
            np.mean(isolated_values)
        )

        try:

            _, p = stats.mannwhitneyu(
                repeat_values,
                isolated_values,
                alternative="two-sided"
            )

            result["p_mw"] = p

        except Exception:
            pass

        # Cliff's delta
        comparisons = (
            repeat_values[:, None]
            -
            isolated_values[None, :]
        )

        result["cliffs_delta"] = (
            np.sum(comparisons > 0)
            -
            np.sum(comparisons < 0)
        ) / comparisons.size

    return result


repeat_results = pd.DataFrame([

    group_summary(
        MET,
        "resist"
    ),

    group_summary(
        MET,
        "rec_time"
    ),

    group_summary(
        MET,
        "post_z"
    ),

    group_summary(
        MET,
        "recovery_gain"
    )

])


print(
    "\n================ REPEAT EFFECT ================\n"
)

print(
    repeat_results.round(3).to_string(
        index=False
    )
)


# ============================================================
# 15. STRATIFIED PERMUTATION TEST
# ============================================================

def stratified_permutation(
    df,
    outcome,
    n_perm=5000,
    seed=42
):

    rng = np.random.default_rng(seed)

    work = df[
        [
            "district",
            "repeat",
            outcome
        ]
    ].dropna()

    if (
        work["repeat"].sum() == 0
        or
        work["repeat"].sum() == len(work)
    ):

        return np.nan, np.nan

    observed = (
        work.loc[
            work["repeat"] == 1,
            outcome
        ].mean()
        -
        work.loc[
            work["repeat"] == 0,
            outcome
        ].mean()
    )

    districts = work["district"].unique()

    null = []

    for _ in range(n_perm):

        shuffled = work["repeat"].copy()

        for district in districts:

            idx = work.index[
                work["district"] == district
            ]

            shuffled.loc[idx] = (
                rng.permutation(
                    work.loc[idx, "repeat"].values
                )
            )

        if shuffled.sum() == 0:
            continue

        if shuffled.sum() == len(shuffled):
            continue

        diff = (
            work.loc[
                shuffled == 1,
                outcome
            ].mean()
            -
            work.loc[
                shuffled == 0,
                outcome
            ].mean()
        )

        null.append(diff)


    null = np.asarray(null)

    if len(null) == 0:
        return observed, np.nan

    p = (
        np.mean(
            np.abs(null)
            >=
            abs(observed)
        )
    )

    return observed, p


print(
    "\n================ STRATIFIED PERMUTATION ================\n"
)

for outcome in [
    "resist",
    "post_z",
    "recovery_gain"
]:

    diff, p = stratified_permutation(
        MET,
        outcome
    )

    print(
        f"{outcome:16s} "
        f"difference = {diff:+.3f} "
        f"p = {p:.3f}"
        if pd.notna(p)
        else
        f"{outcome:16s} "
        f"difference = {diff:+.3f} "
        f"p = NA"
    )


# ============================================================
# 16. OLS REGRESSION
# ============================================================

def fit_ols(
    df,
    response
):

    work = df[
        [
            response,
            "severity",
            "duration",
            "repeat",
            "district"
        ]
    ].dropna()

    if len(work) < 10:
        return None

    try:

        model = smf.ols(
            f"{response} ~ severity + duration + repeat + C(district)",
            data=work
        ).fit()

        return model

    except Exception:

        return None


clustered_model = clustered_regression(
    MET,
    "resist ~ severity + duration + repeat + C(district)"
)

if clustered_model is not None:

    print(
        "\n================ YEAR-CLUSTERED REGRESSION ================\n"
    )

    print(
        clustered_model.summary()
    )
    
print(
    "\n================ REGRESSION ================\n"
)

for response in [
    "resist",
    "rec_time",
    "post_z",
    "recovery_gain"
]:

    model = fit_ols(
        MET,
        response
    )

    if model is None:

        print(
            f"\n{response}: insufficient data"
        )

        continue

    term = "repeat"

    ci = model.conf_int().loc[
        term
    ]

    print(
        f"\n{response} ~ "
        f"severity + duration + repeat + district"
    )

    print(
        f"n = {int(model.nobs)}"
    )

    print(
        f"R² = {model.rsquared:.3f}"
    )

    print(
        f"repeat coefficient = "
        f"{model.params[term]:+.3f}"
    )
    
# ============================================================
# BOOTSTRAP GROUP DIFFERENCES
# ============================================================

print("\n================ BOOTSTRAP ================\n")

for response, label in [
    ("resist", "Resistance"),
    ("rec_time", "Recovery time"),
    ("post_z", "Post-drought residual"),
    ("recovery_gain", "Recovery gain")
]:

    repeat_values = MET.loc[
        MET["repeat"] == 1,
        response
    ].dropna()

    isolated_values = MET.loc[
        MET["repeat"] == 0,
        response
    ].dropna()

    boot = bootstrap_difference(
        repeat_values,
        isolated_values,
        n_boot=CFG["n_boot"],
        seed=CFG["random_seed"]
    )

    print(f"{label}:")
    print(
        f"  Difference = {boot['difference']:+.3f}"
    )
    print(
        f"  95% CI = "
        f"[{boot['ci_low']:+.3f}, "
        f"{boot['ci_high']:+.3f}]"
    )
    print(
        f"  Repeat n = {boot['n_repeat']}"
    )
    print(
        f"  Isolated n = {boot['n_isolated']}"
    )

# ============================================================
# 17. ORDER / LEGACY CHECKS
# ============================================================

print(
    "\n================ ORDER / LEGACY CHECK ================\n"
)

checks = [

    ("order", "resist"),
    ("order", "rec_time"),
    ("order", "recovery_gain"),
    ("order", "post_z"),

    ("prior_severity", "resist"),
    ("prior_severity", "rec_time"),
    ("prior_severity", "recovery_gain"),
    ("prior_severity", "post_z"),

    ("pre_z", "resist"),
    ("pre_z", "rec_time"),
    ("pre_z", "recovery_gain"),
    ("pre_z", "post_z")
]


rows = []

for predictor, response in checks:

    work = MET[
        [predictor, response]
    ].dropna()

    if len(work) < 3:
        continue

    rho, p = stats.spearmanr(
        work[predictor],
        work[response]
    )

    rows.append({

        "predictor": predictor,
        "response": response,
        "n": len(work),
        "rho": rho,
        "p": p
    })


order_results = pd.DataFrame(
    rows
)

print(
    order_results.round(3).to_string(
        index=False
    )
)


# ============================================================
# 18. NDVI VALIDATION
# ============================================================

print(
    "\n================ NDVI VALIDATION ================\n"
)

validation = MET[
    [
        "resist",
        "ndvi_resist"
    ]
].dropna()

if len(validation) >= 3:

    rho, p = stats.spearmanr(
        validation["resist"],
        validation["ndvi_resist"]
    )

    print(
        f"GPP vs NDVI resistance: "
        f"rho = {rho:.3f}, "
        f"p = {p:.3f}, "
        f"n = {len(validation)}"
    )

else:

    print(
        "Insufficient paired GPP/NDVI events."
    )


# ============================================================
# 19. SEASONAL ARTEFACT CHECK
# ============================================================

print(
    "\n================ SEASONAL CHECK ================\n"
)

season_groups = [

    g["resist"].dropna().values

    for _, g
    in MET.groupby("season")

    if g["resist"].notna().sum() >= 2
]

if len(season_groups) >= 2:

    h, p = stats.kruskal(
        *season_groups
    )

    print(
        f"Kruskal-Wallis resistance by season: "
        f"H = {h:.3f}, p = {p:.3f}"
    )

else:

    print(
        "Insufficient seasonal groups."
    )


# ============================================================
# 20. DISTRICT-LEVEL SUMMARY
# ============================================================

print(
    "\n================ DISTRICT SUMMARY ================\n"
)

district_summary = (
    MET
    .groupby("district")
    .agg(
        events=("order", "size"),
        repeat_events=("repeat", "sum"),
        mean_resistance=("resist", "mean"),
        median_resistance=("resist", "median"),
        mean_recovery_time=("rec_time", "mean"),
        recovered_24m=("recovered", "sum"),
        gpp_loss_events=("had_loss", "sum")
    )
)

print(
    district_summary.round(3)
)


# ============================================================
# 21. PLOT 1:
# SPEI + GPP RESPONSE TIMELINE
# ============================================================

fig, axes = plt.subplots(
    12,
    1,
    figsize=(14, 16),
    sharex=False
)

for i, (district, (m, events)) in enumerate(
    ANOM.items()
):

    ax1 = axes[2 * i]
    ax2 = axes[2 * i + 1]

    # SPEI
    ax1.plot(
        m.index,
        m["spei"],
        linewidth=1
    )

    ax1.axhline(
        CFG["spei_threshold"],
        linestyle="--",
        linewidth=1
    )

    ax1.set_ylabel(
        "SPEI-03"
    )

    ax1.set_title(
        district,
        loc="left",
        fontweight="bold"
    )

    # GPP ratio
    ax2.plot(
        m.index,
        m["gpp_rel"],
        linewidth=1
    )

    ax2.axhline(
        1.0,
        linestyle="--",
        linewidth=1
    )

    ax2.axhline(
        CFG["recovery_ratio"],
        linestyle=":",
        linewidth=1
    )

    ax2.set_ylabel(
        "GPP / baseline"
    )

    # Shade droughts
    for _, event in events.iterrows():

        ax1.axvspan(
            event["start"],
            event["end"],
            alpha=0.2
        )

        ax2.axvspan(
            event["start"],
            event["end"],
            alpha=0.2
        )


plt.tight_layout()

plt.savefig(
    "01_drought_gpp_timeline_corrected.png",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 22. PLOT 2:
# RESISTANCE BY REPEAT STATUS
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

plot_data = []

for label, value in [
    ("Isolated / first", 0),
    ("Repeated after incomplete recovery", 1)
]:

    vals = MET.loc[
        MET["repeat"] == value,
        "resist"
    ].dropna()

    for v in vals:

        plot_data.append({
            "group": label,
            "resistance": v
        })


plot_df = pd.DataFrame(
    plot_data
)

if not plot_df.empty:

    plot_df.boxplot(
        column="resistance",
        by="group",
        ax=ax
    )

    ax.axhline(
        CFG["recovery_ratio"],
        linestyle="--",
        linewidth=1,
        label="90% baseline"
    )

    ax.set_xlabel("")
    ax.set_ylabel(
        "Resistance\n(minimum GPP / baseline)"
    )

    ax.set_title(
        "Resistance by drought history"
    )

    plt.suptitle("")

    plt.tight_layout()

    plt.savefig(
        "02_resistance_repeat_vs_isolated.png",
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 23. PLOT 3:
# RECOVERY TIME
# ============================================================

recovery_plot = MET[
    MET["recovered"] == 1
].copy()

if len(recovery_plot) > 0:

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    recovery_plot.boxplot(
        column="rec_time",
        by="repeat",
        ax=ax
    )

    ax.set_xlabel(
        "Repeat status "
        "(0 = isolated/first, "
        "1 = preceded by incomplete recovery)"
    )

    ax.set_ylabel(
        "Recovery time (months)"
    )

    ax.set_title(
        "Observed recovery time"
    )

    plt.suptitle("")

    plt.tight_layout()

    plt.savefig(
        "03_recovery_time.png",
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 24. PLOT 4:
# RESISTANCE VS DROUGHT SEVERITY
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

for district, group in MET.groupby(
    "district"
):

    ax.scatter(
        group["severity"],
        group["resist"],
        label=district,
        alpha=0.8
    )


ax.axhline(
    CFG["recovery_ratio"],
    linestyle="--",
    linewidth=1
)

ax.set_xlabel(
    "Drought severity"
)

ax.set_ylabel(
    "Resistance\n(minimum GPP / baseline)"
)

ax.set_title(
    "Resistance vs drought severity"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    "04_resistance_vs_severity.png",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 25. PLOT 5:
# RESISTANCE VS EVENT ORDER
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 5)
)

for district, group in MET.groupby(
    "district"
):

    ax.scatter(
        group["order"],
        group["resist"],
        label=district,
        alpha=0.8
    )


ax.axhline(
    CFG["recovery_ratio"],
    linestyle="--",
    linewidth=1
)

ax.set_xlabel(
    "Drought order within district"
)

ax.set_ylabel(
    "Resistance"
)

ax.set_title(
    "Resistance across successive drought order"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    "05_resistance_vs_event_order.png",
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 26. FINAL SUMMARY
# ============================================================

print(
    "\n================ FINAL SUMMARY ================\n"
)

print(
    f"Total events: {len(MET)}"
)

print(
    f"Repeated events "
    f"(preceded by incomplete recovery): "
    f"{int(MET['repeat'].sum())}"
)

print(
    f"First/isolated events: "
    f"{int((MET['repeat'] == 0).sum())}"
)

print(
    f"Events recovered: "
    f"{MET['recovery_date'].notna().sum()}"
)

print(
    f"Events not recovered within "
    f"{CFG['max_recovery_months']} months: "
    f"{MET['recovery_date'].isna().sum()}"
)


print(
    f"Districts: "
    f"{MET['district'].nunique()}"
)


# Resistance comparison
r = group_summary(
    MET,
    "resist"
)

print("\nResistance:")

print(
    f"Repeat - isolated = "
    f"{r['difference']:+.3f}"
)

print(
    f"Mann-Whitney p = "
    f"{r['p_mw']:.3f}"
    if pd.notna(r["p_mw"])
    else
    "Mann-Whitney p = NA"
)


# Recovery summary
recovered = MET[
    MET["recovered"] == 1
]["rec_time"].dropna()

print(
    "\nRecovery:"
)

if len(recovered):

    print(
        f"Events recovered within "
        f"{CFG['max_recovery_months']} months: "
        f"{len(recovered)} / {len(MET)}"
    )

    print(
        f"Median recovery time: "
        f"{recovered.median():.1f} months"
    )

    print(
        f"Mean recovery time: "
        f"{recovered.mean():.1f} months"
    )

else:

    print(
        "No events had observable recovery."
    )


print(
    "\nInterpretation:"
)

print(
    "Use 'associated with' rather than "
    "'causes' or 'drives'."
)

print(
    "A non-significant result means that "
    "the dataset did not provide detectable "
    "evidence of an effect; it does not prove "
    "that repeated drought has no effect."
)

print(
    "\nImportant:"
)

print(
    "Drought events in nearby districts may "
    "share the same climate episodes, so "
    "they are not fully independent."
)


# ============================================================
# 27. SAVE MAIN OUTPUTS
# ============================================================

MET.to_csv(
    "drought_event_metrics_corrected.csv",
    index=False
)

district_summary.to_csv(
    "district_summary_corrected.csv"
)

repeat_results.to_csv(
    "repeat_effect_results_corrected.csv",
    index=False
)

order_results.to_csv(
    "order_legacy_results_corrected.csv",
    index=False
)

print(
    "\nSaved:"
)

print(
    "  drought_event_metrics_corrected.csv"
)

print(
    "  district_summary_corrected.csv"
)

print(
    "  repeat_effect_results_corrected.csv"
)

print(
    "  order_legacy_results_corrected.csv"
)

print(
    "  01_drought_gpp_timeline_corrected.png"
)

print(
    "  02_resistance_repeat_vs_isolated.png"
)

print(
    "  03_recovery_time.png"
)

print(
    "  04_resistance_vs_severity.png"
)

print(
    "  05_resistance_vs_event_order.png"
)
import os
import glob
import numpy as np
import pandas as pd
import fastavro
import time as _time
from collections import defaultdict
from datetime import time, timedelta

import neurokit2 as nk

from scipy.signal import butter, filtfilt

import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from dateutil import tz
from dateutil.tz import gettz

import matplotlib.dates as mdates
import itertools

plt.style.use('default')



# --- Default parameters ---
DEFAULT_FS = {
    'accelerometer': 64.00555419921875,
    'eda': 4.003623008728027,
    'bvp': 64.0,
    'steps': 0.20000000298023224,
    'temperature': 1.0     
}

DEFAULT_IMUPARAMS_ACC = {
    'physicalMin': -16,
    'physicalMax': 16,
    'digitalMin': -32768,
    'digitalMax': 32768
}

# --- 1. Read a single Avro file and extract rawData ---
AVRO_READ_TIMEOUTS = []
AVRO_READ_ERRORS = []


def parse_avro_file(file_path, *, max_retries=3, backoff_s=1.0, log_timeout=True):
    """
    Open an Avro file and return the 'rawData' dict for the first record.
    first record because there is only one record per file.

    On cloud-synced or network volumes, files that are not locally cached can raise
    TimeoutError (Errno 60). Corrupt/partial downloads can also raise EOFError
    or "cannot read header" ValueError. We retry those, then skip if still failing.
    """
    try:
        if os.path.getsize(file_path) == 0:
            AVRO_READ_ERRORS.append((file_path, 'zero_size'))
            print(f"Empty avro file; skipping: {file_path}")
            return None
    except OSError:
        # If stat fails, let the read attempt handle it
        pass

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            with open(file_path, 'rb') as f:
                reader = fastavro.reader(f)
                for record in reader:
                    rawData = record.get('rawData')
                    return rawData
            return None
        except (EOFError, ValueError) as e:
            last_err = e
        except TimeoutError as e:
            last_err = e
        except OSError as e:
            if getattr(e, 'errno', None) != 60:
                raise
            last_err = e

        if attempt < max_retries:
            _time.sleep(backoff_s * (2 ** attempt))
            continue

        # Final failure: log and skip
        if isinstance(last_err, (EOFError, ValueError)):
            AVRO_READ_ERRORS.append((file_path, str(last_err)))
            print(f"Bad avro; skipping: {file_path} ({last_err})")
        else:
            if log_timeout:
                AVRO_READ_TIMEOUTS.append(file_path)
                print(f"Timeout reading avro; skipping: {file_path} ({last_err})")
        return None

# --- 2. Merge generic sensor time series, preserving chunks ---
def merge_sensor_timeseries(chunks, default_fs=0):
    """
    Given a list of dicts each with 'timestampStart', 'samplingFrequency', and 'values',
    return two numpy arrays: times (in seconds) and values, preserving absolute times.
    Use default_fs if samplingFrequency is missing or non-positive.
    """
    chunks = sorted(chunks, key=lambda c: c['timestampStart'])
    all_times, all_vals = [], []
    for c in chunks:
        fs = c.get('samplingFrequency', default_fs) or default_fs
        if fs <= 0:
            fs = default_fs
        t0 = c['timestampStart'] / 1e6
        vals = np.array(c.get('values', []), dtype=float)
        if vals.size == 0:
            continue
        times = t0 + np.arange(vals.size) / fs
        all_times.append(times)
        all_vals.append(vals)
    if not all_times:
        return np.array([]), np.array([])
    return np.concatenate(all_times), np.concatenate(all_vals)

# --- 3. Merge accelerometer data (three axes) ---
def merge_acc_timeseries(chunks, default_fs=0):
    """
    Merge accelerometer chunks with x/y/z and timestampStart/samplingFrequency.
    Use default_fs if samplingFrequency is missing or non-positive.
    """
    chunks = sorted(chunks, key=lambda c: c['timestampStart'])
    all_t, all_x, all_y, all_z = [], [], [], []
    for c in chunks:
        fs = c.get('samplingFrequency', default_fs) or default_fs
        if fs <= 0:
            fs = default_fs
        t0 = c['timestampStart'] / 1e6
        x = np.array(c.get('x', []), dtype=float)
        if x.size == 0:
            continue
        y = np.array(c.get('y', []), dtype=float)
        z = np.array(c.get('z', []), dtype=float)
        times = t0 + np.arange(x.size) / fs
        all_t.append(times)
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)
    if not all_t:
        return np.array([]), np.array([]), np.array([]), np.array([])
    return (np.concatenate(all_t),
            np.concatenate(all_x),
            np.concatenate(all_y),
            np.concatenate(all_z))

# --- 4. Merge systolic peaks timestamps ---
def merge_peaks_times(chunks):
    all_ns = []
    for c in chunks:
        all_ns.extend(c.get('peaksTimeNanos', []))
    if not all_ns:
        return np.array([])
    all_ns = np.array(sorted(all_ns), dtype=float)
    return all_ns / 1e9  # convert nanoseconds to seconds

# --- 5. DataFrame builders ---
def df_from_timeseries(times, values):
    """
    Build a Pandas DataFrame for times and values without resampling.
    """
    if times.size == 0:
        return pd.DataFrame()
    idx = (pd.to_datetime(times, unit='s', origin='unix', utc=True).tz_convert('America/New_York'))

    return pd.DataFrame({'value': values}, index=idx)


def df_from_acc_timeseries(times, x, y, z, imuParams=None):
    """
    Build a DataFrame for accelerometer axes in g-units.
    Apply default IMU params if not provided.
    """
    if times.size == 0:
        return pd.DataFrame()
    params = {**DEFAULT_IMUPARAMS_ACC, **(imuParams or {})}
    dp = params['physicalMax'] - params['physicalMin']
    dd = params['digitalMax']  - params['digitalMin']
    scale = dp / dd
    x_g = np.array(x, dtype=float) * scale
    y_g = np.array(y, dtype=float) * scale
    z_g = np.array(z, dtype=float) * scale
    mag = np.sqrt(x_g**2 + y_g**2 + z_g**2)
    idx = (pd.to_datetime(times, unit='s', origin='unix', utc=True).tz_convert('America/New_York'))
        
    return pd.DataFrame({
        'acc_x_g': x_g,
        'acc_y_g': y_g,
        'acc_z_g': z_g,
        'acc_mag_g': mag
    }, index=idx)

def _resolve_date_root(date_folder, patient_id):
    """
    Resolve the true root for a date folder.

    Normal case:
        date_folder/raw_data
        date_folder/digital_biomarkers

    Exception case:
        date_folder/<PATIENT_SUBFOLDER>/raw_data
        date_folder/<PATIENT_SUBFOLDER>/digital_biomarkers

    If multiple child folders exist, only choose the one whose name starts
    with the given patient_id.
    """
    # Normal structure
    if (
        os.path.isdir(os.path.join(date_folder, 'raw_data')) and
        os.path.isdir(os.path.join(date_folder, 'digital_biomarkers'))
    ):
        return date_folder

    # Look one level down, but only for the correct patient
    patient_id = str(patient_id)
    matching_children = []

    for sub in sorted(os.listdir(date_folder)):
        candidate = os.path.join(date_folder, sub)
        if not os.path.isdir(candidate):
            continue

        if not sub.startswith(patient_id):
            continue

        has_raw = os.path.isdir(os.path.join(candidate, 'raw_data'))
        has_biomarkers = os.path.isdir(os.path.join(candidate, 'digital_biomarkers'))

        if has_raw:
            matching_children.append(candidate)

    if len(matching_children) == 1:
        return matching_children[0]

    if len(matching_children) > 1:
        raise FileExistsError(
            f"Multiple matching child folders found for patient {patient_id} under {date_folder}: "
            f"{[os.path.basename(x) for x in matching_children]}"
        )

    raise FileNotFoundError(
        f"No matching child folder starting with patient_id={patient_id} "
        f"and containing raw_data found under {date_folder}"
    )
    
# --- 6. Process a single day folder ---
def process_date_folder(date_folder, *, patient_id, date_label=None, show_progress=True, verbose=False):
    """
    Given a date folder path (containing raw_data/v6/*.avro),
    returns a dict of DataFrames per sensor.

    Handles the exception where an extra layer exists between the date folder
    and raw_data/digital_biomarkers, and only selects the child folder that
    starts with the current patient_id.
    """
    root = _resolve_date_root(date_folder, patient_id)

    raw_dir = os.path.join(root, 'raw_data', 'v6')
    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(f"raw_data/v6 not found in {root}")

    avro_files = sorted(f for f in os.listdir(raw_dir) if f.endswith('.avro'))
    if verbose:
        label = date_label or os.path.basename(date_folder.rstrip(os.sep))
        print(f"Processing date {label}: {len(avro_files)} avro files")

    raw_chunks = []
    label = date_label or os.path.basename(date_folder.rstrip(os.sep))
    file_iter = _progress_iter(avro_files, desc=f"{label} avro", show_progress=show_progress, leave=False)

    for f in file_iter:
        _progress_set_postfix(file_iter, f)
        if verbose and not hasattr(file_iter, "set_postfix_str"):
            print(f"  reading {f}")
        raw_chunks.append(parse_avro_file(os.path.join(raw_dir, f)))

    dfs = {}

    # Accelerometer
    acc_chunks = [r['accelerometer'] for r in raw_chunks if r and 'accelerometer' in r]
    if acc_chunks:
        t, x, y, z = merge_acc_timeseries(acc_chunks, DEFAULT_FS['accelerometer'])
        dfs['accelerometer'] = df_from_acc_timeseries(t, x, y, z, acc_chunks[0].get('imuParams'))

    # EDA
    eda_chunks = [r['eda'] for r in raw_chunks if r and 'eda' in r]
    if eda_chunks:
        t, v = merge_sensor_timeseries(eda_chunks, DEFAULT_FS['eda'])
        dfs['eda'] = df_from_timeseries(t, v)

    # Temperature
    temp_chunks = [r['temperature'] for r in raw_chunks if r and 'temperature' in r]
    if temp_chunks:
        t, v = merge_sensor_timeseries(temp_chunks, DEFAULT_FS['temperature'])
        dfs['temperature'] = df_from_timeseries(t, v)

    # BVP
    bvp_chunks = [r['bvp'] for r in raw_chunks if r and 'bvp' in r]
    if bvp_chunks:
        t, v = merge_sensor_timeseries(bvp_chunks, DEFAULT_FS['bvp'])
        dfs['bvp'] = df_from_timeseries(t, v)

    # Systolic Peaks
    peaks = [r['systolicPeaks'] for r in raw_chunks if r and 'systolicPeaks' in r]
    if peaks:
        t = merge_peaks_times(peaks)
        dfs['systolicPeaks'] = pd.DataFrame(
            {'peaks_s': t},
            index=pd.to_datetime(t, unit='s', origin='unix')
        )

    # Steps
    step_chunks = [r['steps'] for r in raw_chunks if r and 'steps' in r]
    if step_chunks:
        t, v = merge_sensor_timeseries(step_chunks, DEFAULT_FS['steps'])
        dfs['steps'] = df_from_timeseries(t, v)

    # Sleep data
    sleep_dir = os.path.join(root, 'digital_biomarkers', 'aggregated_per_minute')

    if os.path.isdir(sleep_dir):
        for fname in sorted(os.listdir(sleep_dir)):
            if fname.endswith('_sleep-detection.csv'):
                path = os.path.join(sleep_dir, fname)
                sleep = pd.read_csv(path)

                sleep['timestamp_iso'] = pd.to_datetime(
                    sleep['timestamp_iso'], utc=True
                ).dt.tz_convert('America/New_York')

                sleep.set_index('timestamp_iso', inplace=True)

                dfs['sleep_detection'] = sleep[[
                    'sleep_detection_stage',
                    'missing_value_reason'
                ]]
                break

    return dfs

# --- 9. Deduplicate sleep data logic when two files of same date/time exist.
def _resolve_sleep_dup_group(group):
    """
    Given a DataFrame `group` of rows sharing the same timestamp
    in the sleep_detection df, pick the single “correct” row.
    """
    # sanity check—only call this on sleep data
    assert 'sleep_detection_stage' in group.columns, \
        "Can only resolve duplicates for sleep_detection"

    stages = group['sleep_detection_stage']
    missing_codes = {0, 101, 102, 300}

    # 1) prefer any non-missing stage
    real = group.loc[~stages.isin(missing_codes)]
    if not real.empty:
        return real.iloc[0]

    # 2) any 'device…'?
    device_mask = stages.astype(str).str.startswith('device')
    if device_mask.any():
        return group.loc[device_mask].iloc[0]

    # 3) fallback: pick the first
    return group.iloc[0]

def _dedupe_sleep_df(df):
    # If there are no true duplicates, return the sorted df and exit.
    if not df.index.duplicated(keep=False).any():
        return df.sort_index()

    # Otherwise, perform the existing dedupe logic
    is_dup = df.index.duplicated(keep=False)
    uniques = df.loc[~is_dup]
    dupes   = df.loc[ is_dup ]

    tmp = dupes.reset_index()
    idx_col = tmp.columns[0] 
    resolved = []
    for _, group in tmp.groupby(idx_col, sort=False):
        resolved.append(_resolve_sleep_dup_group(group))

    out = pd.DataFrame(resolved)
    out = out.set_index(idx_col)

    return pd.concat([uniques, out]).sort_index()


# --- 7a. Progress helpers ---
def _progress_iter(items, *, desc=None, show_progress=True, leave=False):
    if show_progress:
        try:
            from tqdm.auto import tqdm
            return tqdm(items, desc=desc, leave=leave)
        except Exception:
            pass
    return items


def _progress_set_postfix(it, text):
    try:
        it.set_postfix_str(text)
    except Exception:
        pass


# --- 7b. Cache helpers for per-date persistence ---
def _cache_date_key(date_path):
    return os.path.basename(date_path.rstrip(os.sep))


def _cache_file(cache_dir, date_key, sensor):
    return os.path.join(cache_dir, f"{date_key}__{sensor}.pkl.gz")


def _cache_done_file(cache_dir, date_key):
    return os.path.join(cache_dir, f"{date_key}__.done")


def _save_cached_daily(cache_dir, date_key, daily):
    os.makedirs(cache_dir, exist_ok=True)
    for sensor, df in daily.items():
        path = _cache_file(cache_dir, date_key, sensor)
        df.to_pickle(path, compression='gzip')
    # mark date as complete
    done_path = _cache_done_file(cache_dir, date_key)
    with open(done_path, 'w') as f:
        f.write('ok')


def _load_cached_daily(cache_dir, date_key, sensors=None):
    daily = {}
    if sensors is not None:
        for sensor in sensors:
            path = _cache_file(cache_dir, date_key, sensor)
            if os.path.exists(path):
                daily[sensor] = pd.read_pickle(path, compression='gzip')
        return daily
    # load all cached sensors for the date
    pattern = os.path.join(cache_dir, f"{date_key}__*.pkl.gz")
    for path in glob.glob(pattern):
        sensor = os.path.basename(path).split('__', 1)[1].replace('.pkl.gz','')
        daily[sensor] = pd.read_pickle(path, compression='gzip')
    return daily


# --- 8. Process entire patient folder (all dates) ---
def process_patient_folder(
    patient_longitudinal_folder,
    *,
    patient_id,
    sensors=None,
    cache_dir=None,
    use_cache=True,
    save_cache=True,
    show_progress=True,
    verbose=True
):
    """
    Iterate over each date-subfolder and stitch together all days for one patient.
    Returns a dict of merged DataFrames per sensor across all dates.
    """
    sensor_accum = defaultdict(list)

    date_paths = [os.path.join(patient_longitudinal_folder, d)
                  for d in sorted(os.listdir(patient_longitudinal_folder))]
    date_paths = [p for p in date_paths if os.path.isdir(p)]
    date_iter = _progress_iter(date_paths, desc="Dates", show_progress=show_progress, leave=True)

    for date_path in date_iter:
        date_key = _cache_date_key(date_path)

        if verbose:
            _progress_set_postfix(date_iter, date_key)
            if not hasattr(date_iter, "set_postfix_str"):
                print(f"Processing date {date_key}")

        if cache_dir and use_cache:
            done_path = _cache_done_file(cache_dir, date_key)
            if os.path.exists(done_path):
                if verbose:
                    print(f"Using cache for {date_key}")
                daily = _load_cached_daily(cache_dir, date_key, sensors=sensors)
                if daily:
                    for sensor, df in daily.items():
                        if not df.empty:
                            sensor_accum[sensor].append(df)
                    continue

        try:
            daily = process_date_folder(
                date_path,
                patient_id=patient_id,
                date_label=date_key,
                show_progress=show_progress,
                verbose=verbose
            )
            if sensors is not None:
                daily = {k: v for k, v in daily.items() if k in sensors}
            if cache_dir and save_cache and daily:
                _save_cached_daily(cache_dir, date_key, daily)

        except FileNotFoundError:
            continue

        for sensor, df in daily.items():
            if not df.empty:
                sensor_accum[sensor].append(df)

    merged = {}
    for sensor, dfs in sensor_accum.items():
        concatenated = pd.concat(dfs)
        if sensor == 'sleep_detection':
            merged[sensor] = _dedupe_sleep_df(concatenated)
        else:
            deduped = concatenated[~concatenated.index.duplicated(keep='first')]
            merged[sensor] = deduped.sort_index()

    return merged



# Add patient diary
def read_patient_diary(path_to_excel, pat_id):
    """
    Reads the sheet whose name begins with the given patient ID.
    and returns it as a DataFrame.
    """
    xls = pd.ExcelFile(path_to_excel)
    sheet_name = next(
        (s for s in xls.sheet_names if s.startswith(f"{pat_id},")),
        None
    )
    if sheet_name is None:
        raise ValueError(f"No sheet found for pat_id={pat_id!r}")
    return pd.read_excel(xls, sheet_name=sheet_name, skiprows=1)


def summarize_sleep_windows(
    sleep_df: pd.DataFrame,
    normal_start: time = time(20, 0),
    normal_end:   time = time(9, 0),
    min_night_duration: timedelta = timedelta(hours=3),
    max_start_gap: timedelta = timedelta(hours=1),
):
    """
    Summarize sleep across 'normal' and 'outside' windows for a single-patient DataFrame,
    using dynamic night start based on the first substantial sleep cluster.

    Parameters
    ----------
    sleep_df : pd.DataFrame
        DateTimeIndex + 'sleep_detection_stage' column.
    normal_start : datetime.time
        Start of normal window each day (e.g. 20:00).
    normal_end : datetime.time
        End of normal window (e.g. 09:00 next day).
    min_night_duration : timedelta
        Minimum duration to consider a cluster the main nighttime sleep.
    max_start_gap : timedelta
        Max gap between bed events to merge into the same sleep cluster when
        determining the start of the night.

    Returns
    -------
    pd.DataFrame
        One row per window with:
          - window_type        : 'normal'/'outside'
          - window_start       : Timestamp
          - window_end         : Timestamp
          - sleep_start        : first timestamp of main sleep cluster or NaT
          - sleep_end          : last in-bed timestamp or NaT
          - hours_101          : hours in stage 101
          - hours_102          : hours in stage 102
          - hours_300          : hours in stage 300
          - number_disruptions : count of bed->0->bed events
          - hours_disruption   : total hours spent in those disruptions
          - first_disruption   : Timestamp of first in-bed→0→in-bed or NaT
          - reason             : why row has NaNs ('no 101 sleep') or ''
    """
    STAGES_IN_BED = {101, 102, 300}
    df = sleep_df.sort_index()
    start_ts, end_ts = df.index.min(), df.index.max()

    # Helper to convert a time-of-day into an offset
    def offset(t: time) -> pd.Timedelta:
        return pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)

    # Build calendar-based windows (preserve tz)
    day0 = start_ts.normalize()
    dayN = (end_ts + timedelta(days=1)).normalize()
    windows = []
    # head outside
    windows.append(("outside", day0, day0 + offset(normal_start)))
    cur = day0
    while cur < dayN:
        start_norm = cur + offset(normal_start)
        end_norm = (cur + offset(normal_end)) if normal_end > normal_start else (cur + timedelta(days=1) + offset(normal_end))
        windows.append(("normal", start_norm, end_norm))
        windows.append(("outside", end_norm, start_norm + timedelta(days=1)))
        cur += timedelta(days=1)

    records = []
    for label, w_start, w_end in windows:
        # skip windows outside data
        if w_end <= start_ts or w_start >= end_ts:
            continue
        win = df.loc[w_start : w_end]
        rec = {"window_type": label, "window_start": w_start, "window_end": w_end, "reason": ""}

        # get all bed event timestamps
        bed_times = win.index[win.sleep_detection_stage.isin(STAGES_IN_BED)]
        if bed_times.empty:
            rec.update({
                "sleep_start":        pd.NaT,
                "sleep_end":          pd.NaT,
                "hours_101":          0.0,
                "hours_102":          0.0,
                "hours_300":          0.0,
                "number_disruptions": 0,
                "hours_disruption":   0.0,
                "first_disruption":   pd.NaT,
                "reason":             "no 101 sleep",
            })
            records.append(rec)
            continue

        # 1) cluster bed_times by max_start_gap
        clusters = []
        current = [bed_times[0]]
        for ts in bed_times[1:]:
            if ts - current[-1] <= max_start_gap:
                current.append(ts)
            else:
                clusters.append(current)
                current = [ts]
        clusters.append(current)

        # 2) pick first cluster ≥ min_night_duration, else longest
        main = next((c for c in clusters if (c[-1] - c[0]) >= min_night_duration), None)
        if main is None:
            main = max(clusters, key=lambda c: c[-1] - c[0])

        sleep_start = main[0]
        sleep_end = bed_times.max()
        core = win.loc[sleep_start : sleep_end]

        # 3) sum hours per stage for the core segment
        diffs = core.index.to_series().diff().shift(-1).fillna(timedelta(0))
        sums = diffs.groupby(core.sleep_detection_stage).sum().div(timedelta(hours=1))
        h101, h102, h300 = sums.get(101, 0.0), sums.get(102, 0.0), sums.get(300, 0.0)

        # 4) detect disruptions: bed->0->bed
        in_bed = core.sleep_detection_stage.isin(STAGES_IN_BED)
        prev_bed = in_bed.shift(1, fill_value=False)
        is_zero = core.sleep_detection_stage.eq(0)
        candidates = prev_bed & is_zero
        suffix_bed = in_bed[::-1].cummax()[::-1]
        disruption_mask = candidates & suffix_bed

        if disruption_mask.any():
            times = core.index[disruption_mask]
            number_disruptions = int(disruption_mask.sum())
            hours_disruption = diffs[disruption_mask].sum() / timedelta(hours=1)
            first_disruption = times.min()
        else:
            number_disruptions = 0
            hours_disruption = 0.0
            first_disruption = pd.NaT

        rec.update({
            "sleep_start":        sleep_start,
            "sleep_end":          sleep_end,
            "hours_101":          h101,
            "hours_102":          h102,
            "hours_300":          h300,
            "number_disruptions": number_disruptions,
            "hours_disruption":   hours_disruption,
            "first_disruption":   first_disruption,
        })
        records.append(rec)

    return pd.DataFrame(records).reset_index(drop=True)


def plot_all_sleep_windows(sleep_df, summary_df):
    """
    Loop over each 'normal' window in summary_df and plot the detailed sleep stages 
    between sleep_start and sleep_end, with x-axis in EST timezone.
    
    Parameters
    ----------
    sleep_df : pd.DataFrame
        Original DataFrame with DateTimeIndex and 'sleep_detection_stage' column.
    summary_df : pd.DataFrame
        Output of summarize_sleep_windows(), containing at least:
          - window_type
          - sleep_start
          - sleep_end
    """
    eastern = gettz('America/New_York')
    
    # Filter for normal windows with valid sleep_start/end
    normals = summary_df[
        (summary_df['window_type'] == 'normal') &
        summary_df['sleep_start'].notna()
    ].reset_index(drop=True)
    
    if normals.empty:
        print("No normal sleep windows to plot.")
        return

    for i, row in normals.iterrows():
        start = row['sleep_start']
        end   = row['sleep_end']
        night_label = start.strftime("%Y-%m-%d")
        print(f"Plotting night {i+1} ({night_label}): {start} → {end}")
        
        # Build contiguous segments of identical stage
        window = sleep_df.loc[start:end]
        if window.empty:
            continue
        
        segments = []
        prev_stage = window.iloc[0]['sleep_detection_stage']
        seg_start = window.index[0]
        for ts, rec in window.iloc[1:].iterrows():
            stage = rec['sleep_detection_stage']
            if stage != prev_stage:
                seg_end = ts
                segments.append((seg_start, seg_end - seg_start, prev_stage))
                seg_start = ts
                prev_stage = stage
        # final segment
        final_end = window.index[-1] + (window.index.to_series().diff().iloc[1] 
                                        if len(window) > 1 else pd.Timedelta(minutes=1))
        segments.append((seg_start, final_end - seg_start, prev_stage))

        # Color mapping
        stage_colors = {
            0:   'tab:red',   # Awake
            101: 'tab:green',   # Pure Sleep
            102: 'tab:blue',  # Light Sleep
            300: 'tab:orange',    # Disruption
        }

        # Plot
        fig, ax = plt.subplots(figsize=(10, 2))
        for seg in segments:
            ax.broken_barh(
                [(seg[0], seg[1])], 
                (0, 1),
                facecolors=stage_colors.get(seg[2], 'black'),
                edgecolors='none'
            )
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlim(start, end)
        ax.set_xlabel('Time (EST)')
        ax.set_title(f"Night {i+1} ({night_label}): {start.strftime('%H:%M')} → {end.strftime('%H:%M')}")

        # Set x-axis to EST
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=eastern))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=eastern))

        # Legend
        legend_items = [plt.Line2D([0], [0], color=col, lw=6) for col in stage_colors.values()]
        ax.legend(legend_items, [
            'Awake (0)',
            'Pure Sleep (101)',
            'Light Sleep (102)',
            'Disruption (300)'
        ], bbox_to_anchor=(1.02, 1), loc='upper left')
        
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
        


pat_diary_cols_to_focus_on = ['PT_ID', 'period_id', 'Night', 'Night date begin','0 = no pain next day; 1 = non-migraine pain; 2 = migraine; 3 = nonheadache pain']

def sleep_summary_w_pat_diary_status(pat_diary, sleep_summary_df, pat_diary_cols_to_focus_on):
    pat_diary_only_status = pat_diary[pat_diary_cols_to_focus_on]

    sleep_summary_df['sleep_end_date'] = pd.to_datetime(sleep_summary_df['sleep_end']).dt.date
    sleep_summary_df_normal = sleep_summary_df[
        (sleep_summary_df['window_type'] == 'normal') &
        (sleep_summary_df['reason'] != 'no 101 sleep')
    ]
    pat_diary_only_status['next_day_date'] = pd.to_datetime(pat_diary_only_status['Night date begin']) + pd.Timedelta(days=1)
    pat_diary_only_status['next_day_date'] = pd.to_datetime(pat_diary_only_status['next_day_date']).dt.date

    sleep_summary_df_w_diary = (
        sleep_summary_df_normal
        .merge(
            pat_diary_only_status,
            how='left',
            left_on='sleep_end_date',
            right_on='next_day_date',
            suffixes=('', '_diary')
        )
    )
    return sleep_summary_df_w_diary


def _fill_small_gaps(mask: pd.Series, max_gap_s: float) -> pd.Series:
    """
    Given a boolean mask (True = artifact/missing, False = valid), fill in any
    interior False-run (gap) shorter than max_gap_s seconds if it's bounded
    on both sides by True.
    """
    values = mask.values.copy()
    times = mask.index
    run_id = np.zeros(len(values), dtype=int)
    for i in range(1, len(values)):
        run_id[i] = run_id[i-1] + (values[i] != values[i-1])
    for gid in np.unique(run_id):
        idx = np.where(run_id == gid)[0]
        # this run is a gap (False)
        if not values[idx[0]]:
            start_t = times[idx[0]]
            end_t = times[idx[-1]]
            duration = (end_t - start_t).total_seconds()
            # If bounded by artifacts on both sides and short enough --> fill it 
            if idx[0] > 0 and idx[-1] < len(values) - 1:
                if values[idx[0] - 1] and values[idx[-1] + 1] and duration <= max_gap_s:
                    values[idx] = True
    return pd.Series(values, index=mask.index)

def clean_eda(
    eda_series: pd.Series,
    sleep_summary_df: pd.DataFrame,
    sleep_df: pd.DataFrame,
    sampling_rate: float = 4.0,
    use_hampel: bool = False,
    hampel_window: str = "30s",
    hampel_n_sigmas: float = 3.0,
    use_chebyshev: bool = True,
    cheb_coverage: float = 0.95,
    short_artifact_duration: str = "5s",
    mask_close_gaps: str = "5s"
) -> pd.Series:
    """
    Clean EDA by:
      1) Masking outside 'normal' windows OR where stage != 101
      2) Masking statistical outliers (Chebyshev)
      3) Interpolating short gaps
      4) Filling small valid gaps between remaining NaNs
    """
    # Mask outside normal windows and mask stage != 101
    is_in_window = pd.Series(False, index=eda_series.index)
    for _, row in sleep_summary_df.iterrows():
        if row['window_type'] == 'normal' and pd.notna(row['sleep_start']):
            is_in_window[row['window_start']:row['window_end']] = True

    eda_df = eda_series.to_frame('eda')
    stage = pd.merge_asof(
        eda_df.sort_index(),
        sleep_df[['sleep_detection_stage']].sort_index(),
        left_index=True, right_index=True,
        direction='backward'
    )['sleep_detection_stage']
    is_stage101 = stage == 101

    artifact_mask = ~(is_in_window & is_stage101)

    # Statistical outliers
    rolling_med = eda_series.rolling(hampel_window, center=True).median()
    rolling_mad = (eda_series - rolling_med).abs().rolling(hampel_window, center=True).median()
    if use_hampel:
        artifact_mask |= (eda_series - rolling_med).abs() > (hampel_n_sigmas * rolling_mad)
    if use_chebyshev:
        k_cheb = np.sqrt(1.0 / (1.0 - cheb_coverage))
        artifact_mask |= (rolling_mad > 0) & ((eda_series - rolling_med).abs() > (k_cheb * rolling_mad))

    # Mask and interpolate short NaN gaps
    cleaned = eda_series.copy()
    cleaned[artifact_mask] = np.nan
    limit = int(sampling_rate * pd.Timedelta(short_artifact_duration).total_seconds())
    cleaned = cleaned.interpolate(method='time', limit=limit)

    # Fill small valid gaps between remaining NaNs
    if mask_close_gaps:
        max_gap_s = pd.Timedelta(mask_close_gaps).total_seconds()
        nan_mask = cleaned.isna()
        nan_mask_filled = _fill_small_gaps(nan_mask, max_gap_s)
        cleaned[nan_mask_filled] = np.nan

    return cleaned


def plot_one_night(raw_eda: pd.Series,
                clean_eda: pd.Series,
                start: pd.Timestamp,
                end: pd.Timestamp,
                title: str = None):
    """
    Plot raw vs. cleaned EDA for a single sleep window.
    """
    raw_seg = raw_eda.loc[start:end]
    clean_seg = clean_eda.loc[start:end]

    export_df = pd.DataFrame({
                    'timestamp': raw_seg.index,
                    'raw_eda':   raw_seg.values,
                    'clean_eda': clean_seg.values
                })
    # export_df.to_csv('raw_vs_clean_EDA.csv')
        
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(raw_seg.index, raw_seg.values, label="Raw EDA", alpha=0.6)
    ax.plot(clean_seg.index, clean_seg.values, label="Cleaned EDA", lw=1.5)
    ax.axvspan(start, end, color='lightgrey', alpha=0.3, label="Sleep Window")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Sleep Window: {start.date()}")
    ax.set_xlabel("Time")
    ax.set_ylabel("EDA (µS)")
    ax.legend(loc='upper right')
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


def plot_multiple_nights(raw_eda: pd.Series,
                        clean_eda: pd.Series,
                        sleep_summary_df: pd.DataFrame,
                        night_indices: list[int] = None):
    """
    Plot raw vs. cleaned EDA in separate figures for one or more nights.

    Parameters
    ----------
    raw_eda : pd.Series
        Full-night raw EDA.
    clean_eda : pd.Series
        Full-night cleaned EDA.
    sleep_summary_df : pd.DataFrame
        Must contain 'window_type', 'window_start', 'window_end'.
    night_indices : list[int], optional
        Row indices in sleep_summary_df to plot. If None, plot all 'normal' windows.
    """
    df = sleep_summary_df.reset_index(drop=True)
    if night_indices is None:
        # plot all normal windows
        night_indices = df[(df['window_type'] == 'normal') & ~df['sleep_start'].isna()].index.tolist()

    for idx in night_indices:
        row = df.loc[idx]
        if row['window_type'] != 'normal':
            continue
        start, end = row['sleep_start'], row['sleep_end']
        title = f"Night {idx} ({start.date()})"
        plot_one_night(raw_eda, clean_eda, start, end, title=title)



def detrend_median(series: pd.Series, window: str = "10min") -> pd.Series:
    """Remove slow trend via moving median baseline subtraction."""
    baseline = series.rolling(window=window, center=True, min_periods=1).median()
    return series - baseline


def detrend_butter(series: pd.Series, cutoff: float = 0.005, fs: float = 4.0, order: int = 2) -> pd.Series:
    """High-pass Butterworth filter to remove slow drift."""
    b, a = butter(order, cutoff, btype='highpass', fs=fs)
    filtered = filtfilt(b, a, series.values)
    return pd.Series(filtered, index=series.index)


def plot_eda_temp_and_steps_with_detrends(
    sleep_df: pd.DataFrame,
    eda_series: pd.Series,
    temp_series: pd.Series,
    steps_series: pd.Series,
    summary_df: pd.DataFrame,
    median_window: str = "10min",
    butter_cutoff: float = 0.005,
    fs: float = 4.0
):
    eastern = gettz('America/New_York')
    normals = summary_df[
        (summary_df['window_type'] == 'normal') &
        (summary_df['sleep_start'].notna())
    ].reset_index(drop=True)

    colors = {0:'red', 101:'green', 102:'blue', 300:'orange'}

    for idx, row in normals.iterrows():
        start, end = row['sleep_start'], row['sleep_end']
        label = start.strftime('%Y-%m-%d')

        sleep_win = sleep_df.loc[start:end]
        eda_win   = eda_series.loc[start:end]
        temp_win  = temp_series.loc[start:end]
        steps_win = steps_series.loc[start:end]

        # skip if any is empty
        if sleep_win.empty or eda_win.empty or temp_win.empty or steps_win.empty:
            continue

        # detrends
        eda_med = detrend_median(eda_win, window=median_window)
        eda_but = detrend_butter(eda_win, cutoff=butter_cutoff, fs=fs)

        # build sleep-stage segments
        segments = []
        prev_stage = sleep_win['sleep_detection_stage'].iloc[0]
        seg_start = sleep_win.index[0]
        for ts, rec in sleep_win.iloc[1:].iterrows():
            stage = rec['sleep_detection_stage']
            if stage != prev_stage:
                segments.append((seg_start, ts - seg_start, prev_stage))
                seg_start, prev_stage = ts, stage
        last_delta = (
            sleep_win.index.to_series().diff().iloc[1]
            if len(sleep_win) > 1 else pd.Timedelta(minutes=1)
        )
        segments.append((seg_start, sleep_win.index[-1] + last_delta - seg_start, prev_stage))

        # set up 3-row figure
        fig, axs = plt.subplots(3, 1, sharex=False, figsize=(12, 9))
        titles = ['Raw EDA, Temp & Steps', 'Median-Detrended EDA', 'Butterworth-Detrended EDA']
        data_list = [(eda_win, temp_win, steps_win), (eda_med, None, None), (eda_but, None, None)]

        for i, (ax, title, (primary, secondary, tertiary)) in enumerate(zip(axs, titles, data_list)):
            ax.set_title(f"Night {idx+1} {label}: {title}")
            ax.set_xlim(start, end)

            y_min, y_max = primary.min(), primary.max()
            pad = (y_max - y_min) * 0.05 if y_max > y_min else 1
            ax.set_ylim(y_min - pad, y_max + pad)

            # sleep-stage bars
            for s0, dur, st in segments:
                ax.broken_barh(
                    [(s0, dur)],
                    (y_min - pad, (y_max + pad) - (y_min - pad)),
                    facecolors=colors.get(st), edgecolors='none'
                )

            # x-axis formatting
            if i < 2:
                ax.xaxis.set_ticks_position('top')
                ax.xaxis.set_label_position('top')
                ax.tick_params(axis='x', which='both', labelbottom=False, labeltop=True)
                ax.set_xlabel('')
            else:
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(tz=eastern))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=eastern))
                ax.set_xlabel('Time (EST)')

        # ——— 1) Raw EDA + Temp + Steps ———
        ax0 = axs[0]
        # EDA
        ax0.plot(eda_win.index, eda_win.values, color='yellow', label='EDA Raw')
        ax0.set_ylabel('EDA (µS)')
        ax0.tick_params(axis='y')
        # Temp
        ax_temp = ax0.twinx()
        ax_temp.plot(temp_win.index, temp_win.values, color='grey', label='Temp (°C)')
        ax_temp.set_ylabel('Temp (°C)')
        ax_temp.tick_params(axis='y')
        # Steps (offset further right)
        ax_steps = ax0.twinx()
        ax_steps.spines["right"].set_position(("axes", 1.12))
        ax_steps.plot(steps_win.index, steps_win.values, color='black', label='Steps')
        ax_steps.set_ylabel('Steps')
        ax_steps.tick_params(axis='y')
        # combined legend
        l0, lab0 = ax0.get_legend_handles_labels()
        l1, lab1 = ax_temp.get_legend_handles_labels()
        l2, lab2 = ax_steps.get_legend_handles_labels()
        ax0.legend(l0 + l1 + l2, lab0 + lab1 + lab2, loc='upper right')

        # ——— 2) Median-detrended EDA ———
        axs[1].plot(eda_med.index, eda_med.values, color='magenta', label='Median ΔEDA')
        axs[1].set_ylabel('ΔEDA (µS)')
        axs[1].tick_params(axis='y')
        axs[1].legend(loc='upper right')

        # ——— 3) Butterworth-detrended EDA ———
        axs[2].plot(eda_but.index, eda_but.values, color='cyan', label='Butter ΔEDA')
        axs[2].set_ylabel('ΔEDA (µS)')
        axs[2].tick_params(axis='y')
        axs[2].legend(loc='upper right')

        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()




def detect_original_storms(peaks_index, window_size='1min', min_peaks=5, min_duration='10min'):
    """
    Original definition (Burch):
        - Periods with >= min_peaks per window_size (default 1 minute)
        - Sustained for at least min_duration (default 10 minutes)
    Returns list of (start, end) for each detected storm.
    """
    counts = peaks_index.to_series().groupby(pd.Grouper(freq=window_size)).count()
    flags = counts >= min_peaks
    storms = []
    start_ts = None
    for ts, flag in flags.items():
        if flag and start_ts is None:
            start_ts = ts
        if not flag and start_ts is not None:
            end_ts = ts + pd.Timedelta(window_size)
            if (end_ts - start_ts) >= pd.Timedelta(min_duration):
                storms.append((start_ts, end_ts))
            start_ts = None
    
    if start_ts is not None:
        end_ts = flags.index[-1] + pd.Timedelta(window_size)
        if (end_ts - start_ts) >= pd.Timedelta(min_duration):
            storms.append((start_ts, end_ts))
    
    return storms


def detect_reformed_storms(peaks_index, epoch_seconds=30, min_peaks=3, merge_gap_seconds=300):
    """
    Reformulated definition: [Akane Sano and Rosalind W. Picard, IEEE, 2011]
        - 30-second epochs with >= min_peaks
        - Merge adjacent epochs or those within merge_gap_seconds (default 5 min)
    Returns merged storm intervals.
    """
    if peaks_index.empty:
        return []
    start = peaks_index.min().floor(f"{epoch_seconds}s")
    end = peaks_index.max().ceil(f"{epoch_seconds}s")
    bins = pd.interval_range(start=start, end=end, freq=f"{epoch_seconds}s", closed='left')
    binned = pd.cut(peaks_index, bins)
    counts = pd.Series(peaks_index).groupby(binned).count()
    qualified = [iv for iv, cnt in counts.items() if cnt >= min_peaks]
    storms = []
    if not qualified:
        return storms
    curr_start, curr_end = qualified[0].left, qualified[0].right
    for iv in qualified[1:]:
        if (iv.left - curr_end).total_seconds() <= merge_gap_seconds:
            curr_end = iv.right
        else:
            storms.append((curr_start, curr_end))
            curr_start, curr_end = iv.left, iv.right
    storms.append((curr_start, curr_end))
    return storms


def compute_storm_metrics(storms, peak_times, peaks_df, epoch_seconds=None):
    """
    Compute storm metrics:
        - storm_count: number of storms
        - storm_epochs: total epochs (based on epoch_seconds if provided,
            otherwise based on window granularity in original definition)
        - storm_peaks: total peaks within storms
        - storm_duration_s: total duration in seconds
        - storm_peak_freq_per_min: peaks per minute within storms
        - storm_mean_amplitude: mean SCR amplitude during storms
        - storm_mean_ipi_s: mean inter-peak interval within storms
        - storm_mean_interval_s: mean interval between storm onsets
        - first_storm_onset: timestamp of first storm start
    """
    if not storms:
        return {
            'storm_count': 0,
            'storm_epochs': 0,
            'storm_peaks': 0,
            'storm_duration_s': 0,
            'storm_peak_freq_per_min': np.nan,
            'storm_mean_amplitude': np.nan,
            'storm_mean_ipi_s': np.nan,
            'storm_mean_interval_s': np.nan,
            'first_storm_onset': pd.NaT
        }
    durations, counts, amps, all_ipis = [], [], [], []
    for start, end in storms:
        mask = (peak_times >= start) & (peak_times < end)
        pts = peak_times[mask]
        if pts.empty:
            continue
        durations.append((end - start).total_seconds())
        counts.append(len(pts))
        amps.extend(peaks_df.loc[pts, 'SCR_Amplitude'].tolist())
        # inter-peak intervals within this storm
        ipis = pts.to_series().diff().dt.total_seconds().dropna().tolist()
        all_ipis.extend(ipis)
    total_duration = sum(durations)
    total_peaks = sum(counts)
    freq = total_peaks / (total_duration / 60) if total_duration else np.nan
    
    storm_starts = [start for start, _ in storms]
    print('Storm starts:', storm_starts)
    if len(storm_starts) > 1:
        intervals = [
            (storm_starts[i] - storm_starts[i-1]).total_seconds()
            for i in range(1, len(storm_starts))
        ]
        mean_interval = np.mean(intervals)
    else:
        mean_interval = np.nan
        
    # epoch count based on definition
    epoch_count = int(total_duration / epoch_seconds) if epoch_seconds else int(total_duration / 60)
    return {
        'storm_count': len(storms),
        'storm_epochs': epoch_count,
        'storm_peaks': total_peaks,
        'storm_duration_s': total_duration,
        'storm_peak_freq_per_min': freq,
        'storm_mean_amplitude': np.mean(amps) if amps else np.nan,
        'storm_mean_ipi_s': np.mean(all_ipis) if all_ipis else np.nan,
        'storm_mean_interval_s': mean_interval,
        'first_storm_onset': storms[0][0]
    }


def calculate_peak_and_storm_metrics_new(
    eda_cleaned: pd.Series,
    sleep_summary_df: pd.DataFrame,
    sampling_rate: float = 4.0,
    detrend: str = None,
    detrend_window: str = "10min",
    detrend_cutoff: float = 0.005,
    detrend_order: int = 2,
    # I am adding a threshold for the peaks.
    amplitude_threshold: float = None, 
    # I am also adding a threshold for the distance in seconds between peaks.
    min_peak_distance: float = None
):
    """
    For each 'normal' sleep window, split the cleaned EDA into contiguous non-NaN segments,
    run EDA decomposition & peak detection on each segment, compute segment-level metrics,
    then aggregate to nightly metrics including storms.

    Returns
    -------
    segments_df : pd.DataFrame
        One row per segment with columns:
          ['night_id','segment_start','segment_end','peak_count',
           'mean_amplitude','median_amplitude','max_amplitude','min_amplitude',
           'mean_ipi_s']
    nights_df : pd.DataFrame
        One row per night with aggregated metrics:
          ['night_id','window_start','window_end','total_peak_count',
           'mean_of_means','median_of_medians','min_of_mins','max_of_maxs',
           'mean_ipi_s', ... storm metrics ...]
    """
    # Helper to detrend a segment
    def _detrend(seg):
        if detrend == 'median':
            return detrend_median(seg, window=detrend_window)
        elif detrend == 'butter':
            return detrend_butter(seg, cutoff=detrend_cutoff, fs=sampling_rate, order=detrend_order)
        else:
            return seg

    segments = []
    nights = []

    df = sleep_summary_df.copy()
    if 'night_id' not in df.columns:
        df = df.reset_index().rename(columns={'index':'night_id'})

    for _, row in df.iterrows():
        if row.get('window_type') != 'normal':
            continue
        night_id = row['night_id']
        start, end = row['window_start'], row['window_end']
        window_series = eda_cleaned.loc[start:end]

        # Identify contiguous non-NA segments
        valid = window_series.notna()
        group = (valid != valid.shift()).cumsum()
        seg_metrics = []

        for gid, grp in window_series.groupby(group):
            if not valid.loc[grp.index[0]]:
                continue  # skip NaN runs
            seg = grp.dropna()
            if len(seg) < 2:
                continue  # too short to process

            # Segment duration
            seg_start = seg.index[0]
            seg_end = seg.index[-1]
            seg_duration_m = ((seg_end - seg_start).total_seconds())/60

            # Detrend and process
            seg_proc = _detrend(seg)
            signals, _ = nk.eda_process(seg_proc.values, sampling_rate=sampling_rate)
            signals.index = seg_proc.index
            peaks = signals[signals['SCR_Peaks'] == 1].copy()
            
            if amplitude_threshold is not None:
                peaks = peaks[peaks['SCR_Amplitude'] >= amplitude_threshold]

            if min_peak_distance is not None and not peaks.empty:
                filtered_times = []
                last_time = None
                for t in peaks.index:
                    if last_time is None or (t - last_time).total_seconds() >= min_peak_distance:
                        filtered_times.append(t)
                        last_time = t
                peaks = peaks.loc[filtered_times]
                
            # print(peaks)
            pts = peaks.index

            if pts.empty:
                continue

            # Segment-level metrics
            amp = peaks['SCR_Amplitude']
            mean_amp = amp.mean()
            median_amp = amp.median()
            max_amp = amp.max()
            min_amp = amp.min()
            mean_ipi = pts.to_series().diff().dt.total_seconds().mean()

            segments.append({
                'night_id':           night_id,
                'segment_start':      seg_start,
                'segment_end':        seg_end,
                'segment_duration_m': seg_duration_m,
                'peak_count':         len(peaks),
                'mean_amplitude':     mean_amp,
                'median_amplitude':   median_amp,
                'min_amplitude':      min_amp,
                'max_amplitude':      max_amp,
                'mean_ipi_s':         mean_ipi
            })
            seg_metrics.append({
                'peak_times': pts,
                'amps':       amp,
            })

        # Aggregate nightly metrics
        if not seg_metrics:
            continue

        # Combine peaks across segments
        all_peak_times = pd.Index(np.concatenate([m['peak_times'] for m in seg_metrics]))
        all_amps       = pd.concat([pd.Series(m['amps'].values, index=m['peak_times']) for m in seg_metrics])

        total_peaks = len(all_peak_times)
        mean_of_means    = np.mean([s['mean_amplitude'] for s in segments if s['night_id']==night_id])
        median_of_medians = np.median([s['median_amplitude'] for s in segments if s['night_id']==night_id])
        min_of_mins      = np.min([s['min_amplitude'] for s in segments if s['night_id']==night_id])
        max_of_maxs      = np.max([s['max_amplitude'] for s in segments if s['night_id']==night_id])
        mean_ipi_s       = all_peak_times.to_series().diff().dt.total_seconds().mean()

        # Storm metrics on combined peaks
        orig_intervals = detect_original_storms(all_peak_times)
        reform_intervals = detect_reformed_storms(all_peak_times)
        orig_metrics = compute_storm_metrics(orig_intervals, all_peak_times, pd.DataFrame({'SCR_Amplitude': all_amps}), epoch_seconds=60)
        reform_metrics = compute_storm_metrics(reform_intervals, all_peak_times, pd.DataFrame({'SCR_Amplitude': all_amps}), epoch_seconds=30)

        night_entry = {
            'night_id':           night_id,
            'window_start':       start,
            'window_end':         end,
            'total_peak_count':   total_peaks,
            'mean_of_mean_amplitudes':      mean_of_means,
            'median_of_median_amplitudes':  median_of_medians,
            'min_of_min_amplitudes':        min_of_mins,
            'max_of_max_amplitudes':        max_of_maxs,
            'mean_ipi_s':         mean_ipi_s,
            **{f'orig_{k}': v for k, v in orig_metrics.items()},
            **{f'reform_{k}': v for k, v in reform_metrics.items()}
        }
        nights.append(night_entry)

    segments_df = pd.DataFrame(segments)
    nights_df   = pd.DataFrame(nights)
    
    segments_df['amplitude_threshold'] = amplitude_threshold
    segments_df['min_peak_distance'] = min_peak_distance
    
    nights_df['amplitude_threshold'] = amplitude_threshold
    nights_df['min_peak_distance'] = min_peak_distance
    
    return segments_df, nights_df


def boxplot_by_category(df, value_col, category_col='pain_cat'):
    """
    Plots a boxplot of `value_col` grouped by `category_col`.
    
    Parameters:
    - df: pandas.DataFrame containing the data.
    - value_col: str, name of the numeric column to plot.
    - category_col: str, name of the categorical grouping column.
    """
    data_df = df[[category_col, value_col]].dropna()
    categories = sorted(data_df[category_col].unique())
    
    data = [data_df[data_df[category_col] == cat][value_col] for cat in categories]
    
    fig, ax = plt.subplots()
    ax.boxplot(data, labels=categories)
    ax.set_xlabel(category_col)
    ax.set_ylabel(value_col)
    ax.set_title(f'Boxplot of {value_col} by {category_col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    

amp_vals  = [0, 0.005, 0.01, 0.05]
dist_vals = [0, 1, 3, 5]
def sensitivity_analysis(amp_vals, dist_vals, eda_series_cleaned, sleep_summary_df, sleep_summary_df_w_diary):
    all_segments = []
    all_nights   = []

    for amp, dist in itertools.product(amp_vals, dist_vals):
        seg_df, night_df = calculate_peak_and_storm_metrics_new(
            eda_cleaned=eda_series_cleaned,
            sleep_summary_df=sleep_summary_df,
            detrend='median',
            amplitude_threshold=amp,
            min_peak_distance=dist
        )
        
        all_segments.append(seg_df)
        all_nights.append(night_df)

    sensitivity_analysis_segments_raw_df = pd.concat(all_segments, ignore_index=True)
    sensitivity_analysis_nights_raw_df = pd.concat(all_nights, ignore_index=True)

    sensitivity_analysis_nights_raw_w_diary_df = sleep_summary_df_w_diary.merge(sensitivity_analysis_nights_raw_df, on=['window_start', 'window_end'], how='inner', suffixes=('_1', '_2'))
    sensitivity_analysis_nights_raw_w_diary_df.rename(columns={'0 = no pain next day; 1 = non-migraine pain; 2 = migraine; 3 = nonheadache pain': 'pain_cat'}, inplace=True)
    
    group_map = {0: 'no_pain', 1: 'not_migraine', 2: 'migraine', 3: 'not_migraine'}
    sensitivity_analysis_nights_raw_w_diary_df['pain_group'] = sensitivity_analysis_nights_raw_w_diary_df['pain_cat'].map(group_map)
    return {
        'Segments Sensitivity DF': sensitivity_analysis_segments_raw_df,
        'Nights Sensitivity DF': sensitivity_analysis_nights_raw_df,
        'Nights with Diary Sensitivity DF': sensitivity_analysis_nights_raw_w_diary_df
    }
    
    
def sensitivity_analysis_descriptive(sleep_summary_w_diary_w_peaks_df):
    summary_stat = (
        sleep_summary_w_diary_w_peaks_df
        .groupby('pain_cat')[['total_peak_count',
            'mean_of_mean_amplitudes', 'median_of_median_amplitudes',
            'min_of_min_amplitudes', 'max_of_max_amplitudes', 'mean_ipi_s',
            'orig_storm_count', 'orig_storm_epochs', 'orig_storm_peaks',
            'orig_storm_duration_s', 'orig_storm_peak_freq_per_min',
            'orig_storm_mean_amplitude', 'orig_storm_mean_interval_s',
            'reform_storm_count', 'reform_storm_epochs',
            'reform_storm_peaks', 'reform_storm_duration_s',
            'reform_storm_peak_freq_per_min', 'reform_storm_mean_amplitude',
            'reform_storm_mean_interval_s']]
        .describe()).transpose()
    return summary_stat


metrics = [
    'total_peak_count',
    'median_of_median_amplitudes',
    'mean_ipi_s',
    'orig_storm_count',
    'orig_storm_duration_s',
    'orig_storm_mean_interval_s',
    'reform_storm_count',
    'reform_storm_duration_s',
    'reform_storm_mean_interval_s'
]

def plot_sensitivity_analysis_metrics(metrics, df):
    colors = {
    'no_pain':       'tab:green',
    'not_migraine':  'tab:orange',
    'migraine':      'tab:blue',
    }

    linestyles = {
        1: '-',
        2: '--',
        3: ':',
        4: '-.'
    }

    for metric in metrics:
        fig, ax = plt.subplots()
        for dist, ls in linestyles.items():
            subset = df[df['min_peak_distance']==dist]
            pivot = subset.pivot_table(
                index='amplitude_threshold',
                columns='pain_group',
                values=metric,
                aggfunc='mean'
            )
            for group in pivot.columns:
                ax.plot(
                    pivot.index,
                    pivot[group],
                    marker='o',
                    linestyle=ls,
                    color=colors[group],
                    label=f"{group}, dist={dist}"
                )
        ax.set_xlabel('Amplitude Threshold')
        ax.set_ylabel(metric.replace('_',' ').title())
        ax.set_title(f"Sensitivity of {metric}")
        ax.legend(title='Pain / Dist')
        plt.tight_layout()
        plt.show()
        
        


# Example usage:
#
# data_root = "PATH_TO_DATA"
# participant_id = "PARTICIPANT_ID"
# base_folder = f"{data_root}/{participant_id}/STREAM{participant_id}_LongitudinalData"
# all_sensor_dfs = process_patient_folder(base_folder, patient_id=participant_id)
#
# eda_df = all_sensor_dfs['eda']
# eda_series = eda_df['value']
#
# temp_df = all_sensor_dfs['temperature']
# temp_series = temp_df['value']
#
# steps_series = all_sensor_dfs['steps']
#
# diary_df = read_patient_diary("PATH_TO_DIARY", pat_id=participant_id)
#
# sleep_detection_df = all_sensor_dfs['sleep_detection']
# sleep_summary_df = summarize_sleep_windows(
#     sleep_detection_df,
#     normal_start=time(21, 0),
#     normal_end=time(9, 0),
#     min_night_duration=timedelta(hours=3),
#     max_start_gap=timedelta(hours=1)
# )
#
# sleep_summary_df_w_diary = sleep_summary_w_pat_diary_status(
#     diary_df,
#     sleep_summary_df,
#     pat_diary_cols_to_focus_on
# )
# plot_eda_temp_and_steps_with_detrends(
#     sleep_detection_df,
#     eda_series,
#     temp_series,
#     steps_series,
#     sleep_summary_df
# )
#
# eda_series_cleaned = clean_eda(
#     eda_series,
#     sleep_summary_df,
#     sleep_detection_df,
#     sampling_rate=4.0,
#     use_hampel=False,
#     hampel_window="30s",
#     hampel_n_sigmas=3.0,
#     use_chebyshev=True,
#     cheb_coverage=0.95,
#     short_artifact_duration="3m",
#     mask_close_gaps="10m"
# )
#
# plot_multiple_nights(eda_series, eda_series_cleaned, sleep_summary_df)
# sensitivity_analysis_df = sensitivity_analysis(
#     amp_vals,
#     dist_vals,
#     eda_series_cleaned,
#     sleep_summary_df,
#     sleep_summary_df_w_diary
# )
# sensitivity_analysis_nights_raw_w_diary_df = sensitivity_analysis_df['Nights with Diary Sensitivity DF']

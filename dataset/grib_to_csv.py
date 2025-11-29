#!/usr/bin/env python3
"""
grib_to_csv_pure_fixed.py

Pure-Python GRIB -> CSV extractor that works with various eccodes Python API versions.

- Reads messages one-by-one using the eccodes Python binding
- Groups by shortName, writes small temporary GRIBs per shortName
- Opens temp GRIBs using xarray/cfgrib, converts to DataFrame
- Merges monthly files per site and writes site CSVs in site_csv/
"""
import warnings
from pathlib import Path
import re
import tempfile
import os
import pandas as pd
import xarray as xr
from tqdm import tqdm

# import eccodes module and use functions dynamically (robust to API name differences)
import eccodes as ecc

warnings.filterwarnings("ignore", message=".*edition.*")
warnings.filterwarnings("ignore", category=UserWarning)

INPUT_DIR = Path("cams_downloads_monthly")
OUTPUT_DIR = Path("site_csv")
OUTPUT_DIR.mkdir(exist_ok=True)

# match your filenames like: cams_site_1_2019-01.grib
FNAME_RE = re.compile(r"cams_site_(\d+)_(\d{4})-(\d{2})\.grib$", re.IGNORECASE)


def list_files():
    if not INPUT_DIR.exists():
        print("INPUT_DIR not found:", INPUT_DIR)
        return []
    return sorted([p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".grib"])


def parse_fname(p):
    m = FNAME_RE.match(p.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


# dynamic wrappers for eccodes functions (safe if name differs across versions)
def _codes_new_from_file(fobj):
    # try common names
    for name in ("codes_grib_new_from_file", "codes_new_from_file", "codes_new_from_file_stream"):
        fn = getattr(ecc, name, None)
        if callable(fn):
            try:
                return fn(fobj)
            except Exception:
                # let caller handle None/Exceptions
                raise
    raise RuntimeError("No eccodes 'new_from_file' function found")


def _codes_get(handle, key):
    # typical name is codes_get
    fn = getattr(ecc, "codes_get", None)
    if callable(fn):
        return fn(handle, key)
    # try alternative
    fn2 = getattr(ecc, "get", None)
    if callable(fn2):
        return fn2(handle, key)
    raise RuntimeError("No eccodes 'codes_get' function found")


def _codes_get_message(handle):
    fn = getattr(ecc, "codes_get_message", None)
    if callable(fn):
        return fn(handle)
    # try alternative names (some builds differ)
    for alt in ("get_message", "codes_message_get"):
        fn2 = getattr(ecc, alt, None)
        if callable(fn2):
            return fn2(handle)
    raise RuntimeError("No eccodes 'codes_get_message' function found")


def _codes_handle_delete(handle):
    # try several possible cleanup function names
    for name in ("codes_handle_delete", "codes_release", "codes_handle_free", "codes_close"):
        fn = getattr(ecc, name, None)
        if callable(fn):
            try:
                fn(handle)
            except Exception:
                # ignore errors on deletion
                pass
            return
    # if none found, nothing to do


# read messages grouped by shortName
def extract_messages_by_shortName(path):
    msg_dict = {}
    # open file in binary mode and iterate messages
    with open(path, "rb") as fin:
        while True:
            try:
                h = _codes_new_from_file(fin)
            except Exception:
                break
            if h is None:
                break
            try:
                sn = None
                try:
                    sn = _codes_get(h, "shortName")
                except Exception:
                    sn = None
                # get raw message bytes
                try:
                    msg = _codes_get_message(h)
                except Exception:
                    msg = None
                # free handle
                try:
                    _codes_handle_delete(h)
                except Exception:
                    pass
                if sn and msg:
                    msg_dict.setdefault(sn, []).append(msg)
            except Exception:
                try:
                    _codes_handle_delete(h)
                except Exception:
                    pass
                continue
    return msg_dict


def write_temp_grib(messages, sn):
    # write raw bytes to a temporary file; return path
    tf = tempfile.NamedTemporaryFile(prefix=f"tmp_{sn}_", suffix=".grib", delete=False)
    tf_name = tf.name
    tf.close()
    with open(tf_name, "wb") as out:
        for m in messages:
            out.write(m)
    return Path(tf_name)


def ds_to_df(ds):
    # robustly convert xarray dataset to dataframe
    try:
        df = ds.to_dataframe().reset_index()
    except Exception:
        parts = []
        for v in ds.data_vars:
            try:
                parts.append(ds[v].to_dataframe().reset_index())
            except Exception:
                continue
        if not parts:
            return pd.DataFrame()
        df = parts[0]
        for p in parts[1:]:
            df = pd.merge(df, p, how="outer")
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    return df


def process_single_file(path):
    msg_by_sn = extract_messages_by_shortName(path)
    if not msg_by_sn:
        # final fallback: try opening the file with cfgrib directly (errors ignored)
        try:
            ds = xr.open_dataset(str(path), engine="cfgrib", backend_kwargs={"errors": "ignore"})
            df = ds_to_df(ds)
            ds.close()
            return df
        except Exception:
            return None

    dfs = []
    temp_paths = []
    for sn, msgs in msg_by_sn.items():
        try:
            tp = write_temp_grib(msgs, sn)
            temp_paths.append(tp)
            # open with cfgrib
            try:
                ds = xr.open_dataset(str(tp), engine="cfgrib", backend_kwargs={"errors": "ignore"})
                df_sn = ds_to_df(ds)
                ds.close()
                if df_sn is not None and not df_sn.empty:
                    dfs.append(df_sn)
            except Exception:
                # ignore failure for this shortName
                pass
        except Exception:
            pass

    # cleanup temp files
    for t in temp_paths:
        try:
            t.unlink(missing_ok=True)
        except Exception:
            pass

    if not dfs:
        return None

    # If time exists, merge on time index; else concat across columns
    if any("time" in d.columns for d in dfs):
        dfs_idx = [d.set_index("time") for d in dfs if "time" in d.columns]
        merged = pd.concat(dfs_idx, axis=1, join="outer", sort=True).reset_index()
    else:
        merged = pd.concat(dfs, axis=1, join="outer")

    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged


def main():
    files = list_files()
    if not files:
        print("No GRIB files found in", INPUT_DIR)
        return

    files_by_site = {}
    for f in files:
        parsed = parse_fname(f)
        if parsed is None:
            print("Skipping (pattern mismatch):", f.name)
            continue
        site, year, month = parsed
        files_by_site.setdefault(site, []).append((f, year, month))

    for site in sorted(files_by_site):
        print(f"\nProcessing site {site}")
        rows = []
        for path, year, month in tqdm(files_by_site[site], desc=f"Site {site}", unit="file"):
            df = process_single_file(path)
            if df is None or df.empty:
                print("  WARN: no data from", path.name)
                continue
            df["site"] = site
            df["file_year"] = year
            df["file_month"] = month
            df["source_file"] = path.name
            rows.append(df)

        if not rows:
            print("  No data for site", site)
            continue

        merged = pd.concat(rows, ignore_index=True, sort=False)

        if "time" in merged.columns:
            merged = merged.sort_values("time").drop_duplicates().reset_index(drop=True)
            merged["year"] = merged["time"].dt.year
            merged["month"] = merged["time"].dt.month
            merged["day"] = merged["time"].dt.day
            merged["hour"] = merged["time"].dt.hour

        out = OUTPUT_DIR / f"cams_site_{site}_merged.csv"
        merged.to_csv(out, index=False)
        print("Saved:", out, "rows:", len(merged))


if __name__ == "__main__":
    main()

"""Worklog-33 control D2: copy the first N views of an RNA H5 (values and attributes unchanged).

    python rbq_subset_h5.py --src <h5> --dst <h5> --views 100
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--views", type=int, required=True)
    args = ap.parse_args()
    import h5py

    with h5py.File(args.src, "r") as s, h5py.File(args.dst, "w") as d:
        for k in s.keys():
            d.create_dataset(k, data=s[k][: args.views], dtype=s[k].dtype)
        for k, v in s.attrs.items():
            d.attrs[k] = v
        d.attrs["subset_of"] = args.src
        d.attrs["subset_views"] = args.views
        print({k: d[k].shape for k in d.keys()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

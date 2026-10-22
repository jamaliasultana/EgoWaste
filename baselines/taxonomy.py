#!/usr/bin/env python3
"""
taxonomy.py
======================
Repairs the `material` column in the dataset index.

WHY THIS IS NEEDED
------------------
The meta CSVs store only the fine-grained waste class in `category_name`, with
no material prefix. The dataset builder therefore fell back to using the class
itself as the material, producing ~50 "material" classes instead of the 15 in
the paper's taxonomy. Every model was then trained on the wrong task.

This script maps each waste class to its material category using the taxonomy
from the supplementary material, rewrites `material` in index.csv, and reports
anything it could not match so it can be fixed by hand.

Matching is done on a normalised key (lowercased, punctuation stripped) and
then, for leftovers, by fuzzy match with a report so nothing is silently
mislabelled.

USAGE
  python taxonomy.py --data_dir ./active_obj_data
  python taxonomy.py --data_dir ./active_obj_data --dry_run
"""

import argparse
import difflib
import json
import os
import re
import shutil

import pandas as pd

# taxonomy: material -> fine-grained waste classes (supplementary Tables 1-2)
TAXONOMY = {
    "Plastics": [
        "Black Plastic Containers", "Containers #1/Clamshells", "Containers #2 Colored",
        "Containers #2 Natural", "Containers #3-7 (Not #5)", "Containers #5/PP",
        "Deposit #1 Bottles", "Garbage Bags", "Non-Deposit", "Other #1 Bottles",
        "Other Plastic Film", "Other Plastics", "Other Rigid/Bulky Plastics",
        "Other Unnumbered Containers & Fragments", "Plastic", "Retail Plastic Bags",
        "Rigid Plastic",
    ],
    "Paper": [
        "Boxboard", "Compostable Paper", "Gable Tops", "Hardcover Books",
        "Magazine/Glossy", "Mail", "Old Corrugated Cardboard", "Old Newspaper",
        "Other Paper", "Other Recyclable Paper", "Pizza Boxes",
        "Printed/Office Paper", "Shredded Paper", "Softcover/Paperback Books",
    ],
    "Aluminum": ["Aluminum Foil", "Deposit Aluminum Containers", "Other Aluminum",
                 "Other Aluminum Containers"],
    "Glass": ["Deposit Glass Containers", "Glass Fragments", "Other Glass",
              "Other Glass Containers"],
    "Food": ["Food"],
    "Ferrous": ["Ferrous Containers/Ferris Container", "Other Ferrous"],
    "Other Organics": ["Other Organics"],
    "Other Inorganics": ["Ceramics",
                         "Do-It-Yourself Construction & Demolition Debris",
                         "Other Inorganics"],
    "Rubber and Leather": ["Other Rubber & Leather", "Rubber & Leather"],
    "Yard Waste": ["Yard Waste"],
    "Wood": ["Other Wood", "Wood"],
    "Textiles": ["Clothing", "Other Textiles", "Textiles"],
    "Other Non-Ferrous": ["Other Non-Ferrous"],
    "HHW": ["Household Hazardous Waste"],
    "E-Waste": ["Electronic-Containing Wastes"],
}

# common abbreviations and spelling variants seen in the annotation tool
ALIASES = {
    "occ": "Old Corrugated Cardboard",
    "onp": "Old Newspaper",
    "hc books": "Hardcover Books",
    "pb books": "Softcover/Paperback Books",
    "office paper": "Printed/Office Paper",
    "shred paper": "Shredded Paper",
    "comp paper": "Compostable Paper",
    "compostable compost": "Compostable Paper",
    "other recyc paper": "Other Recyclable Paper",
    "magazine": "Magazine/Glossy",
    "al foil": "Aluminum Foil",
    "dep al cont": "Deposit Aluminum Containers",
    "other al": "Other Aluminum",
    "other al cont": "Other Aluminum Containers",
    "dep glass": "Deposit Glass Containers",
    "glass frag": "Glass Fragments",
    "other glass cont": "Other Glass Containers",
    "dep 1": "Deposit #1 Bottles",
    "deposit 1": "Deposit #1 Bottles",
    "1 clamshell": "Containers #1/Clamshells",
    "clamshell": "Containers #1/Clamshells",
    "2 colored": "Containers #2 Colored",
    "2 natural": "Containers #2 Natural",
    "3 7": "Containers #3-7 (Not #5)",
    "3 to 7": "Containers #3-7 (Not #5)",
    "5 pp": "Containers #5/PP",
    "5": "Containers #5/PP",
    "other 1": "Other #1 Bottles",
    "1 other": "Other #1 Bottles",
    "black cont": "Black Plastic Containers",
    "non dep": "Non-Deposit",
    "nondeposit": "Non-Deposit",
    "plastic film": "Other Plastic Film",
    "rigid bulky": "Other Rigid/Bulky Plastics",
    "other cont": "Other Unnumbered Containers & Fragments",
    "retail bags": "Retail Plastic Bags",
    "ferrous cont": "Ferrous Containers/Ferris Container",
    "ferris container": "Ferrous Containers/Ferris Container",
    "other org": "Other Organics",
    "other inorg": "Other Inorganics",
    "diy c d": "Do-It-Yourself Construction & Demolition Debris",
    "diy c&d": "Do-It-Yourself Construction & Demolition Debris",
    "other r l": "Other Rubber & Leather",
    "r l": "Rubber & Leather",
    "other nf": "Other Non-Ferrous",
    "e waste": "Electronic-Containing Wastes",
    "electronic containing wastes": "Electronic-Containing Wastes",
    "hhw": "Household Hazardous Waste",
}


def norm(s):
    s = str(s).lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_lookup():
    lut = {}
    for mat, classes in TAXONOMY.items():
        for c in classes:
            lut[norm(c)] = (mat, c)
    for alias, target in ALIASES.items():
        for mat, classes in TAXONOMY.items():
            if target in classes:
                lut.setdefault(norm(alias), (mat, target))
                break
    # material names themselves may appear directly
    for mat in TAXONOMY:
        lut.setdefault(norm(mat), (mat, mat))
    lut.setdefault(norm("plastic"), ("Plastics", "Plastic"))
    lut.setdefault(norm("paper"), ("Paper", "Other Paper"))
    return lut


def resolve(value, lut, cutoff=0.86):
    key = norm(value)
    if key in lut:
        return lut[key][0], lut[key][1], "exact"
    hit = difflib.get_close_matches(key, list(lut), n=1, cutoff=cutoff)
    if hit:
        m, c = lut[hit[0]]
        return m, c, f"fuzzy:{hit[0]}"
    return None, None, "unmatched"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./active_obj_data")
    ap.add_argument("--index", default=None)
    ap.add_argument("--cutoff", type=float, default=0.86)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    idx_path = args.index or os.path.join(args.data_dir, "index.csv")
    df = pd.read_csv(idx_path)
    lut = build_lookup()

    src_col = "waste_class" if "waste_class" in df.columns else "material"
    uniq = sorted(df[src_col].astype(str).unique())

    mapping, report = {}, []
    for v in uniq:
        mat, cls, how = resolve(v, lut, args.cutoff)
        mapping[v] = (mat, cls, how)
        report.append({"raw": v, "material": mat, "waste_class": cls,
                       "match": how, "count": int((df[src_col] == v).sum())})

    rep = pd.DataFrame(report).sort_values("count", ascending=False)
    print("=" * 78)
    print("LABEL MAPPING")
    print("=" * 78)
    print(rep.to_string(index=False))

    bad = rep[rep["material"].isna()]
    fuzzy = rep[rep["match"].str.startswith("fuzzy")]
    if len(fuzzy):
        print(f"\n[check] {len(fuzzy)} value(s) matched only fuzzily -- verify these:")
        print(fuzzy[["raw", "waste_class", "count"]].to_string(index=False))
    if len(bad):
        print(f"\n[!] {len(bad)} value(s) could not be mapped "
              f"({int(bad['count'].sum())} samples). Add them to ALIASES:")
        print(bad[["raw", "count"]].to_string(index=False))

    df["_material_new"] = df[src_col].map(lambda v: mapping[str(v)][0])
    df["_class_new"] = df[src_col].map(lambda v: mapping[str(v)][1])

    keep = df["_material_new"].notna()
    print(f"\nmapped {int(keep.sum())}/{len(df)} samples "
          f"into {df.loc[keep, '_material_new'].nunique()} material categories")
    print(df.loc[keep, "_material_new"].value_counts().to_string())

    if args.dry_run:
        print("\n--dry_run: nothing written")
        return

    shutil.copyfile(idx_path, idx_path + ".bak")
    out = df[keep].copy()
    out["waste_class"] = out["_class_new"]
    out["material"] = out["_material_new"]
    out = out.drop(columns=["_material_new", "_class_new"])
    out.to_csv(idx_path, index=False)

    with open(os.path.join(args.data_dir, "label_mapping.json"), "w") as f:
        json.dump({k: {"material": v[0], "waste_class": v[1], "match": v[2]}
                   for k, v in mapping.items()}, f, indent=2)

    print(f"\nbackup   -> {idx_path}.bak")
    print(f"rewrote  -> {idx_path}")
    print(f"mapping  -> {os.path.join(args.data_dir, 'label_mapping.json')}")
    print("\nRe-run training; --label_level material should now report 15 classes.")


if __name__ == "__main__":
    main()

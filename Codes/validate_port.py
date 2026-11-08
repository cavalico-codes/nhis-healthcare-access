import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from nhis_survey import *
from thesis_tables import parse_tables

def label_to_key(var, label):
    left = label.replace("–", "-").split(" vs ")[0].strip()
    inv = {v: k for k, v in LEVEL_LABELS[var].items()}
    return inv[left]

def compare(thesis_rows, res, tag):
    r = res.set_index(["var", "level"])
    diffs = []
    for row in thesis_rows:
        k = (row["var"], label_to_key(row["var"], row["label"]))
        if k not in r.index:
            diffs.append((row["var"], row["label"], None, None, None)); continue
        m = r.loc[k]
        diffs.append((row["var"], row["label"], m.OR - row["OR"], m.lo - row["lo"], m.hi - row["hi"], m.p, row["p"]))
    dd = pd.DataFrame(diffs, columns=["var", "label", "dOR", "dlo", "dhi", "p_py", "p_thesis"])
    print(f"{tag}: rows={len(dd)}  max|dOR|={dd.dOR.abs().max():.4f}  max|dlo|={dd.dlo.abs().max():.4f}  max|dhi|={dd.dhi.abs().max():.4f}")
    return dd

if __name__ == "__main__":
    T = parse_tables()
    d = recode(load_adult())
    jobs = {2: ("HINOTYR_A", 2, None), 6: ("MEDDL12M_A", 2, None), 7: ("MEDNG12M_A", 2, None),
            8: ("TRANSPOR_A", 2, None), 9: ("RXDG12M_A", 2, None)}
    for tno, (y, ev, _) in jobs.items():
        for dist in ("t", "z"):
            res, info = svy_logit(d, y, ev, dist=dist)
            print(f"Table {tno} [{y} event={ev}] dist={dist} n={info['n']} events={info['n_events']} df={info['df']}")
            compare(T[tno], res, f"   vs thesis Table {tno}")

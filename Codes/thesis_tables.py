"""Parse the printed regression tables (Tables 2-9) out of the thesis .docx so the Python port can be
checked against what the SAS run originally produced."""
import re, docx
from pathlib import Path
DOCX = Path(__file__).resolve().parent.parent / "CarlOlson_FinalProject_HTM692Course.docx"
TABLE_IDX = {2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11}
VAR_MAP = {"POV_CAT": "POV_CAT", "FDSCAT4_A": "FDSCAT4_A", "HOUTENURE_A": "HOUTENURE_A", "REGION": "REGION",
           "URBRRL23": "URBRRL23", "SEX_A": "SEX_A", "EDU_CAT": "EDU_CAT", "RACE_CAT": "RACE_CAT",
           "AGE_CAT": "AGE_CAT", "PHSTAT_A": "PHSTAT_A"}
def _num(s):
    s = s.strip().replace("<", "").replace(">", "")
    try: return float(s)
    except: return None
def parse_tables():
    d = docx.Document(DOCX); out = {}
    for tno, idx in TABLE_IDX.items():
        rows, var = [], None
        for r in d.tables[idx].rows[2:]:
            cells = []
            for c in r.cells:                       # merged cells repeat -> dedupe consecutive identical objects
                if not cells or c._tc is not cells[-1]._tc: cells.append(c)
            txt = [c.text.strip() for c in cells]
            if not any(txt): continue
            if txt[0] in VAR_MAP: var = txt[0]; txt = txt[1:]
            elif txt[0] == "": txt = txt[1:]
            if len(txt) < 3 or var is None: continue
            label = txt[0]; nums = [_num(x) for x in txt[1:] if x not in ("", "–", "—", "-")]
            nums = [x for x in nums if x is not None]
            if len(nums) < 4: continue
            rows.append(dict(var=var, label=label, OR=nums[0], lo=nums[1], hi=nums[2], p=nums[3]))
        out[tno] = rows
    return out
if __name__ == "__main__":
    t = parse_tables()
    for k, v in t.items(): print(k, len(v), v[0], v[-1])

"""Controlled negative controls, not natural model failures or cross-level repairs."""
import json
from copy import deepcopy
from pathlib import Path

from core.cover_cabto_api import FineCover
from core.cover_cabto_grounding import CONFIG, save
from run_cover_cabto_method import execute

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/cover_cabto_method"


def main():
    candidates=json.loads((OUT/"session_candidates.json").read_text())
    model=candidates["models"][0]
    good=candidates["programs"]["pick_shrimp"]
    stale=deepcopy(model);stale["del"].remove("at_source(shrimp)")
    cases=[("return_true_without_motion",model,"def policy(api):\n    return True\n"),
           ("remove_lift_from_code",model,good.replace("    api.lift()\n","")),
           ("omit_source_delete_in_model",stale,good)]
    rows=[]
    for name,h,source in cases:
        fine=FineCover(json.loads(CONFIG.read_text()),render=False)
        try:
            rec=execute(fine,h,source)
            rows.append({"case":name,"provenance":"controlled_negative_not_LM_sample","rejected":not rec["effect_ok"],"evaluation":rec})
            print(name,rows[-1]["rejected"],rec["violations"])
        finally:fine.close()
    save(OUT/"negative_controls.json",{"all_rejected":all(r["rejected"] for r in rows),"cases":rows})
    assert all(r["rejected"] for r in rows)


if __name__=="__main__":main()

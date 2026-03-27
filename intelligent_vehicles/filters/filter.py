from typing import List, Dict
import math
import time

class Filter:
    @staticmethod
    def filter_to_ego_prediction_step(preds, ego_ts_ms, pr_frequency, pr_sampling, min_points: int = 10) -> List[Dict]:
        out = []

        for p in preds:
            cur_location = p["cur_location"]
            pr  = p["prediction"]
            ts0 = p["pred_ts_ms"]         # source start (ms)
            t   = pr["t"]                 # seconds
            xy  = pr["xy"]
            cov = pr["cov"]

            ego_pred_ts_ms = ego_ts_ms + int(round(1000.0 * (1/pr_frequency)))
            ego_gate_tms   = ego_pred_ts_ms + int(round(1000.0 * (1/pr_sampling)))
            abs_t = [ts0 + int(round(1000.0 * s)) for s in t]
            filtered = [(tms, pt, cv) for tms, pt, cv in zip(abs_t, xy, cov) if tms >= ego_gate_tms]
            
            if len(filtered) < min_points:
                continue
        
            # rebase future to start at its first kept timestamp
            new_t   = [round((tms-ego_pred_ts_ms) / 1000.0, 3) for tms, _, _ in filtered]
            new_xy  = [pt for _, pt, _ in filtered]
            new_cov = [cv for _, _, cv in filtered]
            
            # cur_location should be chosen the closest to ego_ts_ms (cur_location corresponds to ts0)
            candidates = [(-1, ts0)] + [(i, tms) for i, tms in enumerate(abs_t)]
            best_idx, _ = min(candidates, key=lambda p: abs(ego_ts_ms - p[1]))
            
            if best_idx != -1:
                # keep full 7D box if we have it; only update x,y
                new_x, new_y = xy[best_idx]
                if hasattr(cur_location, "__len__") and len(cur_location) >= 3:
                    cl = list(cur_location)
                    cl[0], cl[1] = new_x, new_y
                    cur_location = cl
                else:
                    # fallback: if it was already 2D, keep it 2D
                    cur_location = [new_x, new_y]

            out.append({
                "id": p["id"],
                "category": p["category"],
                "cur_location": cur_location,
                "pred_ts_ms": ego_pred_ts_ms,
                "prediction": {
                    "t": new_t,
                    "xy": new_xy,
                    "cov": new_cov,
                }
            })
    
        return out

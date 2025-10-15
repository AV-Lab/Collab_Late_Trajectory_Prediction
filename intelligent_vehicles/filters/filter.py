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
            t   = pr["t"]                        # seconds
            xy  = pr["xy"]
            cov = pr["cov"]


            # For predictions 
            ego_next_pr_ts_ms = ego_ts_ms + int(round(1000.0 * (1/pr_frequency))) # when next prediction will run 
            ego_gate_tms = ego_next_pr_ts_ms + int(round(1000.0 * (1/pr_sampling))) 
            abs_t = [ts0 + int(round(1000.0 * s)) for s in t]
            filtered = [(tms, pt, cv) for tms, pt, cv in zip(abs_t, xy, cov) if tms >= ego_gate_tms]
            
            if len(filtered) < min_points: continue

            # rebase future to start at its first kept timestamp
            new_t   = [round((tms-ego_next_pr_ts_ms) / 1000.0, 3) for tms, _, _ in filtered]
            new_xy  = [pt for _, pt, _ in filtered]
            new_cov = [cv for _, _, cv in filtered]
            
            # cur_location should be chosen the closest to ego_ts_ms (cur_location corresponds to ts0)
            candidates = [(-1, ts0)] + [(i, t) for i, t in enumerate(abs_t)]
            best_idx, _ = min(candidates, key=lambda p: abs(ego_ts_ms - p[1]))
            
            if best_idx != -1:
                cur_location = xy[best_idx]

            out.append({
                "category": p["category"],
                "cur_location": cur_location,           
                "pred_ts_ms": ego_next_pr_ts_ms,
                "prediction": {
                    "t": new_t,
                    "xy": new_xy,
                    "cov": new_cov,
                }
            })
    
        return out

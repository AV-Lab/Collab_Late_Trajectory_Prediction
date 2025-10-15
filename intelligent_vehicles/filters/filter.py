from typing import List, Dict
import math

class Filter:
    @staticmethod
    def filter_to_ego_prediction_step(preds, ego_ts_ms, sampling, min_points: int = 10) -> List[Dict]:
        out = []

        for p in preds:
            cur_location = p["cur_location"]
            pr  = p["prediction"]
            ts0 = int(pr["pred_ts_ms"])         # source start (ms)
            t   = pr["t"]                        # seconds
            xy  = pr["xy"]
            cov = pr["cov"]


            # For predictions 
            ego_prediction_ts_ms = ego_ts_ms + int(round(1000.0 * (1/sampling)))
            abs_t = [ts0 + int(round(1000.0 * s)) for s in t]
            filtered = [(tt, tms, pt, cv) for tt, tms, pt, cv in zip(t, abs_t, xy, cov) if tms >= ego_prediction_ts_ms]
            shift = (filtered[0][1] - ego_prediction_ts_ms) / 1000.0

            # need enough FUTURE samples to keep this trajectory
            if len(filtered) < min_points: continue

            # rebase future to start at its first kept timestamp
            new_t   = [tt-shift for tt, _, _, _ in filtered]
            new_xy  = [pt for _, _, pt, _ in filtered]
            new_cov = [cv for _, _, _, cv in filtered]
            
            # cur_location should be chosen the closest to ego_ts_ms (cur_location corresponds to ts0)
            candidates = [(-1, ts0)] + [(i, t) for i, t in enumerate(abs_t)]
            best_idx, _ = min(candidates, key=lambda p: abs(ego_ts_ms - p[1]))
            
            if best_idx != -1:
                cur_location = filtered[best_idx][2]

            out.append({
                "category": p["category"],
                "cur_location": cur_location,           
                "prediction": {
                    "pred_ts_ms": ego_prediction_ts_ms,
                    "t": new_t,
                    "xy": new_xy,
                    "cov": new_cov,
                }
            })
    
        return out

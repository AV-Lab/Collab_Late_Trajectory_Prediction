# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import linear_sum_assignment


class ObjectMap:
    def __init__(self):
        self.G = nx.Graph()
        self._tmp_seq = 0
        self.matching_dist = 0.2

    class Node:
        def __init__(
            self,
            category,
            cur_location,
            type_,
            timestamp=None, # last update by ego
            mean_traj=None,
            cov_traj=None,
        ):
            self.category = category
            self.cur_location = cur_location
            self.last_updated = timestamp
            self.future_trajectory = {"pred": mean_traj, "cov": cov_traj}             # Ego / original prediction
            self.fused_trajectory = None             # Fused prediction (e.g., from GP fusion)
            self.type = int(type_) 
            self.pool = []   

        def __repr__(self):
            return (
                f"Node(cat={self.category}, type={self.type}, "
                f"pos={self.cur_location}, "
                f"pred={self.future_trajectory}, "
                f"fused_pred={self.fused_trajectory}, "
                f"pool={self.pool})"
            )

    def _next_tmp_id(self) -> str:
        self._tmp_seq += 1
        return f"tmp_{self._tmp_seq:06d}"

    def add_to_prediction_pool(self, nid, pred):
        """Append a prediction dict to the node's pool."""
        node = self.G.nodes[nid]['node_data']
        node.pool.append(pred)
        self.G.nodes[nid]['node_data'] = node

    def promote_category_II_to_I(self, node_id, cat_2_node_id, category, cur_pos, mt, ct, t):
        """
        Promote an existing Category II node (temp_id) to Category I (new_id),
        migrating its pool.
        """
        
        temp_node = self.G.nodes[cat_2_node_id]['node_data']
        pool_copy = list(temp_node.pool)
        self.remove_node(cat_2_node_id)
        self.add_node_category_I(node_id, category, cur_pos, t, mt, ct)
        self.G.nodes[node_id]['node_data'].pool = pool_copy


    def update_node(self, node_id, cur_pos, t, mean_traj, cov_traj):
        node = self.G.nodes[node_id]['node_data']
        node.cur_location = cur_pos
        node.last_updated = t
        node.future_trajectory = {"pred": mean_traj, "cov": cov_traj}
        # New ego prediction → fused result is now stale; reset it
        node.fused_trajectory = None
        self.G.nodes[node_id]['node_data'] = node

    def add_node_category_I(self, node_id, category, cur_pos, t, mean_traj, cov_traj):
        node = self.Node(category, cur_pos, 1, t, mean_traj, cov_traj)
        self.G.add_node(node_id, node_data=node)

    def add_node_category_II(self, nid, category, loc, pred):
        """
        Create a Category II node (remote/broadcast) and initialize its pool
        with a single prediction dict.
        """
        node = self.Node(category, loc, 2)
        node.pool.append(pred)
        self.G.add_node(nid, node_data=node)

    def remove_node(self, node_id):
        self.G.remove_node(node_id)

    def update_by_predictor(self, tracklets, mean_trajs, cov_trajs, t):
        """
        Update the prediction map using tracker outputs.
        - Updates Category I nodes that already exist (by id).
        - Promotes nearby Category II nodes to Category I when a new tracklet appears.
        - Removes *unmatched Category I* nodes.
        - Adds new Category I nodes for remaining unmatched tracklets.
        """
        cur_nodes = self.G.nodes
        matched = []     
        unmatched = []   

        # Update existing Category I nodes 
        for idx, (tr, mt, ct) in enumerate(zip(tracklets, mean_trajs, cov_trajs)):
            nid = tr["id"]
            if nid in cur_nodes:
                cur_pos = tr["current_pos"]
                self.update_node(nid, cur_pos, t, mt, ct)
                matched.append(nid)
            else:
                unmatched.append(idx)

        # Promote Category II nodes using category-compatible spatial matching.
        matched_cat_2 = set()
        unmatched_categories = sorted({tracklets[idx]["category"] for idx in unmatched})
        for category in unmatched_categories:
            track_indices = [
                idx for idx in unmatched
                if tracklets[idx]["category"] == category
            ]
            cat2_ids = [
                nid for nid in self.G.nodes
                if self.G.nodes[nid]['node_data'].type == 2
                and self.G.nodes[nid]['node_data'].category == category
            ]
            if not track_indices or not cat2_ids:
                continue

            track_positions = np.vstack([
                np.asarray(tracklets[idx]["current_pos"][:2], float)
                for idx in track_indices
            ])
            cat2_positions = np.vstack([
                np.asarray(self.G.nodes[nid]['node_data'].cur_location[:2], float)
                for nid in cat2_ids
            ])
            diff = track_positions[:, None, :] - cat2_positions[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            rows, cols = linear_sum_assignment(dists)

            for r, c in zip(rows, cols):
                if float(dists[r, c]) > self.matching_dist:
                    continue

                track_idx = track_indices[r]
                cat_2_node_id = cat2_ids[c]
                node_id = tracklets[track_idx]["id"]
                cur_pos = tracklets[track_idx]["current_pos"]
                mt = mean_trajs[track_idx]
                ct = cov_trajs[track_idx]
                self.promote_category_II_to_I(
                    node_id,
                    cat_2_node_id,
                    category,
                    cur_pos,
                    mt,
                    ct,
                    t,
                )
                matched_cat_2.add(track_idx)
                matched.append(node_id)
            
        unmatched = [idx for idx in unmatched if idx not in matched_cat_2]

        # Remove unmatched Category I
        matched_set = set(matched)
        for node_id in list(self.G.nodes):
            node = self.G.nodes[node_id]['node_data']
            if node_id not in matched_set:
                if node.type == 1:
                    self.remove_node(node_id)

        # Add new Category I nodes for remaining unmatched tracklets
        for idx in unmatched:
            tr = tracklets[idx]
            node_id = tr["id"]
            category = tr["category"]
            cur_pos = tr["current_pos"]
            mean_traj = mean_trajs[idx]
            cov_traj = cov_trajs[idx]
            self.add_node_category_I(node_id, category, cur_pos, t, mean_traj, cov_traj)
         


    def match_shared_predictions(self, shared_predictions):
        """
        Match map nodes to category-compatible incoming shared predictions by
        minimal Euclidean distance using Hungarian assignment and
        self.matching_dist.

        Args:
            shared_predictions: prediction dictionaries containing canonical
                categories and current object locations.

        Returns:
            matches:           List[(prediction_index, node_id)] accepted within matching_dist
            unmatched_nodes:   List[node_id]
            unmatched_objects: List[int]  (indices into shared_predictions)
        """
        # Fast exits
        if len(self.G.nodes) == 0 or len(shared_predictions) == 0:
            return [], list(self.G.nodes), list(range(len(shared_predictions)))

        node_ids = list(self.G.nodes)
        matches = []
        matched_nodes = set()
        matched_predictions = set()

        node_categories = {
            self.G.nodes[nid]['node_data'].category
            for nid in node_ids
        }
        prediction_categories = {
            prediction["category"]
            for prediction in shared_predictions
        }

        for category in sorted(node_categories & prediction_categories):
            category_node_ids = [
                nid for nid in node_ids
                if self.G.nodes[nid]['node_data'].category == category
            ]
            prediction_indices = [
                idx for idx, prediction in enumerate(shared_predictions)
                if prediction["category"] == category
            ]

            node_positions = np.vstack([
                np.asarray(self.G.nodes[nid]['node_data'].cur_location[:2], float)
                for nid in category_node_ids
            ])
            prediction_positions = np.vstack([
                np.asarray(shared_predictions[idx]["cur_location"][:2], float)
                for idx in prediction_indices
            ])
            diff = node_positions[:, None, :] - prediction_positions[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            rows, cols = linear_sum_assignment(dists)

            for r, c in zip(rows, cols):
                if float(dists[r, c]) > self.matching_dist:
                    continue

                nid = category_node_ids[r]
                prediction_idx = prediction_indices[c]
                matches.append((prediction_idx, nid))
                matched_nodes.add(nid)
                matched_predictions.add(prediction_idx)

        unmatched_nodes = [nid for nid in node_ids if nid not in matched_nodes]
        unmatched_objects = [
            idx for idx in range(len(shared_predictions))
            if idx not in matched_predictions
        ]

        return matches, unmatched_nodes, unmatched_objects

    def add_new_objects(self, ego_location, unmatched_predictions, shared_predictions, max_dist=25):
        """
        Add remote/broadcast objects that aren't matched to local tracks, only if
        they are within `max_dist` meters from ego (2D distance).

        Args:
            unmatched_predictions: iterable of dicts with keys:
                - "category": str
                - "cur_location": [x, y] or np.ndarray shape (2,)
                - "prediction": {"timestamp": float, "pred": {...}, "cov": {...}}
            ego_xy: (x, y) of ego in world frame (meters)
            max_dist: max allowed distance (meters)

        Returns:
            added_ids: list of node_ids that were added (temporary IDs).
        """
        ex, ey = ego_location["x"], ego_location["y"]
        added_ids = []
        

        for idx in unmatched_predictions:
            # robust extraction
            p = shared_predictions[idx]
            category = p["category"]
            loc = p["cur_location"]
            pred = p["prediction"]
            
            if loc is None: continue

            x, y = loc[0], loc[1]

            # distance gate
            if np.hypot(x - ex, y - ey) > max_dist: continue

            # assign a temporary id and add as Category II
            nid = self._next_tmp_id()
            self.add_node_category_II(nid, category, loc, pred)         
            added_ids.append(nid)
        
        print(f"Total added nodes of category II: {len(added_ids)}")
        return added_ids
    
    
    def update_predictions(self, fused_predictions):
        """
        Overwrite nodes' fused_trajectory with fused predictions.
        ``cov`` is ``None`` for deterministic GP output and may contain a
        timestamp-to-covariance mapping for future probabilistic fusers.
        """
        for k, v in fused_predictions.items():
            pred = {"pred": v["pred"], "cov": v["cov"]}
            self.G.nodes[k]['node_data'].fused_trajectory = pred
            
    def extract_predictions(self, category_II_nodes=True):
        """
        Return a list of dicts summarizing each node's current state.
        """
        out = []
        for nid in self.G.nodes:
            node = self.G.nodes[nid]['node_data']
            if not category_II_nodes and node.type == 2:
                continue
            
            out.append({
                "id": nid,
                "category": node.category,
                "cur_location": node.cur_location,
                "timestamp": node.last_updated,
                "prediction": node.future_trajectory,
                "fused_prediction": node.fused_trajectory,
            })
        return out

    def update_pools(self, matches, shared_predictions):
        """
        Append associated remote predictions. The first current-cycle share
        refreshes a persistent Category II node's current state.
        """
        for i, nid in matches:
            node_data = self.G.nodes[nid]['node_data']
            if node_data.type == 2 and not node_data.pool:
                node_data.cur_location = shared_predictions[i]["cur_location"]
            node_data.pool.append(shared_predictions[i]["prediction"])
            self.G.nodes[nid]['node_data'] = node_data

    def extract_pools(self):
        """
        Return a dict:
          {node_id: (last_updated, future_trajectory, pool, node.type)}
        """
        res = {}
        for nid in self.G.nodes:
            node = self.G.nodes[nid]['node_data']
            res[nid] = (node.last_updated, node.future_trajectory, node.pool, node.type)
        return res

    def empty_pools(self):
        for nid in self.G.nodes:
            self.G.nodes[nid]['node_data'].pool = []

    def extract_categories(self):
        """Return semantic object categories without changing the fusion pool tuple."""
        return {nid: self.G.nodes[nid]['node_data'].category for nid in self.G.nodes}

    def remove_unrefreshed_category_II_nodes(self):
        """Remove Category II nodes that received no share this cycle."""
        for nid in list(self.G.nodes):
            node = self.G.nodes[nid]['node_data']
            if node.type == 2 and not node.pool:
                self.remove_node(nid)

    def advance_category_II_nodes(self):
        """Advance surviving Category II anchors to the next prediction step."""
        for nid in self.G.nodes:
            node = self.G.nodes[nid]['node_data']
            if node.type != 2:
                continue

            first_timestamp = sorted(
                node.fused_trajectory["pred"].keys(),
                key=float,
            )[0]
            next_position = node.fused_trajectory["pred"][first_timestamp]
            current_location = list(node.cur_location)
            current_location[0] = next_position[0]
            current_location[1] = next_position[1]
            node.cur_location = current_location

    def reset(self):
        self.G.clear()

    def __repr__(self):
        repr_ = ""
        for nid in self.G.nodes:
            repr_ += f"node_id={nid}, data={self.G.nodes[nid]} \n"
        return repr_

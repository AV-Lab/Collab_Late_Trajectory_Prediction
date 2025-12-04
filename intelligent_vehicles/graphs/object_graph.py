# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import linear_sum_assignment
import logging
logger = logging.getLogger(__name__)


class ObjectGraph:
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
            self.future_trajectory = {"pred": mean_traj, "cov": cov_traj}
            self.type = int(type_) 
            self.pool = []   
            self.life_span = True

        def __repr__(self):
            return (
                f"Node(cat={self.category}, type={self.type}, "
                f"pos={self.cur_location}, "
                f"pred={self.future_trajectory}, "
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
        Update graph using tracker outputs.
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

        # Promoting Category II nodes to Category I for new tracklets
        um_positions = np.array([np.asarray(tracklets[idx]["current_pos"][:2], float) for idx in unmatched])
        cat2_ids = [nid for nid in self.G.nodes if self.G.nodes[nid]['node_data'].type == 2]
        if len(um_positions) > 0 and len(cat2_ids) > 0:
            cat2_positions = np.vstack([np.asarray(self.G.nodes[nid]['node_data'].cur_location[:2], float) for nid in cat2_ids])
            diff = um_positions[:, None, :] - cat2_positions[None, :, :]
            dists = np.linalg.norm(diff, axis=2)
            rows, cols = linear_sum_assignment(dists)

            # track which unmatched got resolved
            matched_cat_2 = set()

            for r, c in zip(rows, cols):
                dist = float(dists[r, c])
                if dist <= self.matching_dist:
                    track_idx = unmatched[r]
                    cat_2_node_id = cat2_ids[c]
                    node_id = tracklets[track_idx]["id"]
                    cur_pos = tracklets[track_idx]["current_pos"]
                    category = tracklets[track_idx]["category"]
                    mt = mean_trajs[track_idx]
                    ct = cov_trajs[track_idx]
                    self.promote_category_II_to_I(node_id, cat_2_node_id, category, cur_pos, mt, ct, t)
                    matched_cat_2.add(track_idx)
                    matched.append(node_id)
            
            #update matched 
            unmatched = [umr for umr in unmatched if umr not in matched_cat_2]

        # Remove unmatched Category I
        matched_set = set(matched)
        for node_id in list(self.G.nodes):
            node = self.G.nodes[node_id]['node_data']
            if node_id not in matched_set:
                if node.type == 1:
                    self.remove_node(node_id)
                elif node.type == 2:
                    if not node.life_span:
                        self.remove_node(node_id)
                    else:
                        node.life_span = False

        # Add new Category I nodes for remaining unmatched tracklets
        for idx in unmatched:
            tr = tracklets[idx]
            node_id = tr["id"]
            category = tr["category"]
            cur_pos = tr["current_pos"]
            mean_traj = mean_trajs[idx]
            cov_traj = cov_trajs[idx]
            self.add_node_category_I(node_id, category, cur_pos, t, mean_traj, cov_traj)
         


    def match_shared_predictions(self, objs_locations):
        """
        Match current graph nodes to incoming object locations by minimal Euclidean distance,
        using Hungarian assignment and thresholding by self.matching_dist.

        Args:
            objs_locations: list of [x, y] (or (x, y)) locations for currently observed objects.

        Returns:
            matches:           List[(obj_index, node_id)] accepted within matching_dist
            unmatched_nodes:   List[node_id]
            unmatched_objects: List[int]  (indices into objs_locations)
        """
        # Fast exits
        if len(self.G.nodes) == 0 or len(objs_locations) == 0:
            return [], list(self.G.nodes), list(range(len(objs_locations)))

        # Collect node ids and their 2D positions
        node_ids = list(self.G.nodes)
        Gpos = np.array([np.asarray(self.G.nodes[nid]['node_data'].cur_location[:2]) for nid in node_ids], dtype=float)
        Opos = np.array([np.asarray(loc[:2]) for loc in objs_locations], dtype=float)  # shape (M, 2)

        # Build (N x M) cost matrix of Euclidean distances
        diff = Gpos[:, None, :] - Opos[None, :, :]
        dists = np.linalg.norm(diff, axis=2)  # (N, M)

        # Hungarian assignment (minimize total distance)
        rows, cols = linear_sum_assignment(dists)

        # Accept only pairs within matching_dist
        matches = []
        matched_nodes = set()
        matched_objs = set()
        for r, c in zip(rows, cols):
            d = float(dists[r, c])
            if d <= self.matching_dist:
                nid = node_ids[r]
                matches.append((int(c), nid))
                matched_nodes.add(nid)
                matched_objs.add(int(c))

        unmatched_nodes = [nid for nid in node_ids if nid not in matched_nodes]
        unmatched_objects = [i for i in range(len(objs_locations)) if i not in matched_objs]

        return matches, unmatched_nodes, unmatched_objects

    def add_new_objects(self, ego_location, unmatched_predictions, shared_predictions, max_dist=50):
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
        
        logger.info(f"Total added nodes of category II: {len(added_ids)}")
        return added_ids
    
    
    def update_predictions(self, fused_predictions):
        """
        Overwrite nodes' future_trajectory with fused predictions.
        fused_predictions: {node_id: {"pred": {t:[x,y]}, "cov": {t:[[2x2]]}}}
        """
        for k, v in fused_predictions.items():
            pred = {"pred": v["pred"], "cov": v["cov"]}
            self.G.nodes[k]['node_data'].future_trajectory = pred
            
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
            })
        return out

    def update_pools(self, matches, shared_predictions):
        """
        For each match (obj_index, node_id), append the corresponding shared
        prediction into the node's pool.
        """
        for i, nid in matches:
            node_data = self.G.nodes[nid]['node_data']
            node_data.pool.append(shared_predictions[i]["prediction"])
            self.G.nodes[nid]['node_data'] = node_data

    def extract_pools(self):
        """
        Return a dict:
          {node_id: (last_updated, future_trajectory, pool)}
        """
        res = {}
        for nid in self.G.nodes:
            node = self.G.nodes[nid]['node_data']
            res[nid] = (node.last_updated, node.future_trajectory, node.pool, node.type)
        return res

    def empty_pools(self):
        for nid in self.G.nodes:
            self.G.nodes[nid]['node_data'].pool = []

    def reset(self):
        self.G.clear()

    def __repr__(self):
        repr_ = ""
        for nid in self.G.nodes:
            repr_ += f"node_id={nid}, data={self.G.nodes[nid]} \n"
        return repr_

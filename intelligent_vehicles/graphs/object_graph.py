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
        
    class Node:
        def __init__(self, 
                     category,
                     cur_location,
                     type_,
                     mean_traj = None,
                     cov_traj = None, 
                     timestamp = None):
            
            self.category = category
            self.cur_location = cur_location
            self.future_trajectory = {"timestamp": timestamp, "pred": mean_traj, "cov": cov_traj}
            self.type = int(type_) # 1 or 2 
            self.pool = []
            
        def __repr__(self):
            return (f"Node(cat={self.category}, type={self.type}, "
                           f"pos={self.cur_location}, "
                           f"pred={self.future_trajectory}, "
                           f"pool={self.pool})")
 
    def _next_tmp_id(self) -> str:
        self._tmp_seq += 1
        return f"tmp_{self._tmp_seq:06d}"

    def update_node(self, node_id, cur_pos, mean_traj, cov_traj, t):
        node = self.G.nodes[node_id]['node_data']
        node.cur_location = cur_pos
        node.future_trajectory = {"timestamp": t, "pred": mean_traj, "cov": cov_traj}
        self.G.nodes[node_id].update(node_data=node)
        
    def add_node_category_I(self, node_id, category, cur_pos, mean_traj, cov_traj, t):
        node = self.Node(category, cur_pos, 1, mean_traj, cov_traj, t)
        self.G.add_node(node_id, node_data=node)

    def add_node_category_II(self, nid, category, loc, pred):
        # For category 2 node add to the pool
        node = self.Node(category, loc, 2)
        node.pool.append(pred)
        self.G.add_node(nid, node_data=node)

    def remove_node(self, node_id):
        self.G.remove_node(node_id)
    
    def update_by_predictor(self, tracklets, mean_trajs, cov_trajs, t):
        cur_nodes = self.G.nodes
        matched = [] # matched
        unmatched = [] # unmatched
        
        # Update exsisting nodes
        for idx, (tr, mt, ct) in enumerate(zip(tracklets, mean_trajs, cov_trajs)):
            node_id = tr["id"]
            if node_id in cur_nodes:
                cur_pos = tr["current_pos"]
                self.update_node(node_id, cur_pos, mt, ct, t)
                matched.append(node_id)
            else:
                unmatched.append((idx, node_id))
        
        # Remove untracked nodes
        matched = set(matched)
        cur_nodes  = set(cur_nodes)
        nodes_to_delete = cur_nodes - matched
        for node_id in nodes_to_delete:
            self.remove_node(node_id)
            
        # Add new nodes         
        for (idx, node_id) in unmatched:
            category = tracklets[idx]["category"]
            cur_pos = tracklets[idx]["current_pos"]
            mean_traj = mean_trajs[idx] 
            cov_traj = cov_trajs[idx]
            self.add_node_category_I(node_id, category, cur_pos, mean_traj, cov_traj, t)
            
    def extract_predictions(self):
        return [{"id": nid,
                 "category": self.G.nodes[nid]['node_data'].category,
                 "cur_location": self.G.nodes[nid]['node_data'].cur_location,
                 "prediction": self.G.nodes[nid]['node_data'].future_trajectory} for nid in self.G.nodes]
    
    def match_shared_predictions(self, objs_locations, max_dist: float = 0.5):
        """
        Match current graph nodes to incoming object locations by minimal Euclidean distance.

        Args:
            objs_locations: list of [x, y] (or (x, y)) locations for currently observed objects.
            max_dist: maximum allowed distance (meters) to accept a match.

        Returns:
            matches:           List[(node_id, obj_index, distance)]
            unmatched_nodes:   List[node_id]
            unmatched_objects: List[int]  (indices into objs_locations)
        """
        # Fast exits
        if len(self.G.nodes) == 0 or len(objs_locations) == 0:
            return [], list(self.G.nodes), list(range(len(objs_locations)))

        # Collect node ids and their 2D positions
        node_ids = list(self.G.nodes)
        Gpos = np.array(
            [np.asarray(self.G.nodes[nid]['node_data'].cur_location)[:2] for nid in node_ids],
            dtype=float
        )
        Opos = np.asarray(objs_locations, dtype=float)  # shape (M, 2)

        # Build (N x M) cost matrix of Euclidean distances
        diff = Gpos[:, None, :] - Opos[None, :, :]
        dists = np.linalg.norm(diff, axis=2)  # (N, M)

        # Hungarian assignment (minimize total distance)
        rows, cols = linear_sum_assignment(dists)

        # Accept only pairs within max_dist
        matches = []
        for r, c in zip(rows, cols):
            d = float(dists[r, c])
            if d <= max_dist:
                nid = node_ids[r]
                matches.append((int(c), nid))

        return matches
    
    def add_new_objects(self, ego_location, unmatched_predictions, max_dist=50):
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

        for p in unmatched_predictions:
            # robust extraction
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
    
    def updtae_predictions(self, fused_predictions):
        for k,v in fused_predictions.items():
            pred = {"timestamp": v["timestamp"], "pred": v["pred"], "cov":v["cov"]}
            self.G.nodes[k]['node_data'].future_trajectory = pred
   
    def update_pools(self, matches, shared_predictions):
        for i, nid in matches:
            self.G.nodes[nid]['node_data'].pool.append(shared_predictions[i]["prediction"])
    
    def extract_pools(self):
        return {nid: (self.G.nodes[nid]['node_data'].future_trajectory, 
                      self.G.nodes[nid]['node_data'].pool) for nid in self.G.nodes}
    
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
        

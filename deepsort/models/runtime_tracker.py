import os

import torch

from typing import List, Dict
from .utils import logits_to_scores
from .motion import Motion
from .reid import ReIDExtractor
from utils.box_ops import box_cxcywh_to_xyxy

from structures.track_instances import TrackInstances


class RuntimeTracker:
    def __init__(self, det_score_thresh: float = 0.7, track_score_thresh: float = 0.6,
                 miss_tolerance: int = 5,
                 use_motion: bool = False, motion_min_length: int = 3, motion_max_length: int = 5,
                 visualize: bool = False, use_dab: bool = True,
                 use_reid: bool = True, reid_thresh: float = 0.4):
        self.det_score_thresh = det_score_thresh
        self.track_score_thresh = track_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0
        self.use_motion = use_motion
        self.visualize = visualize
        self.motion_min_length = motion_min_length
        self.motion_max_length = motion_max_length
        self.motions: Dict[Motion] = {}
        self.use_dab = use_dab
        
        # DeepSORT ReID components
        self.use_reid = use_reid
        self.reid_thresh = reid_thresh
        self.reid_extractor = ReIDExtractor() if use_reid else None
        self.lost_gallery = {}  # {id: {'features': [f1, f2, ...], 'last_box': box, 'time': t}}
        self.max_lost_time = 30  # Frames to keep a lost track in gallery

    def update(self, model_outputs: dict, tracks: List[TrackInstances], image: torch.Tensor = None):
        assert len(tracks) == 1
        model_outputs["scores"] = logits_to_scores(model_outputs["pred_logits"])
        n_dets = len(model_outputs["det_query_embed"])

        if self.visualize:
            os.makedirs("./outputs/visualize_tmp/runtime_tracker/", exist_ok=True)
            visualize_ids = tracks[0].ids.cpu().tolist()

        # Update tracks.
        tracks[0].boxes = model_outputs["pred_bboxes"][0][n_dets:]
        tracks[0].logits = model_outputs["pred_logits"][0][n_dets:]
        tracks[0].output_embed = model_outputs["outputs"][0][n_dets:]
        tracks[0].scores = logits_to_scores(tracks[0].logits)
        
        # Extract features for active tracks if ReID is enabled
        if self.use_reid and image is not None:
            active_indices = [i for i in range(len(tracks[0])) if tracks[0].scores[i][tracks[0].labels[i]] >= self.track_score_thresh]
            if active_indices:
                active_boxes = box_cxcywh_to_xyxy(tracks[0].boxes[active_indices])
                active_features = self.reid_extractor.extract(image, active_boxes)
                # We could store these features to update the gallery later
        
        for i in range(len(tracks[0])):
            obj_id = tracks[0].ids[i].item()
            if tracks[0].scores[i][tracks[0].labels[i]] < self.track_score_thresh:
                tracks[0].disappear_time[i] += 1
            else:
                if self.use_motion and tracks[0].disappear_time[i] > 0:
                    self.motions[obj_id].clear()
                tracks[0].disappear_time[i] = 0
                if self.use_motion:
                    self.motions[obj_id].add_box(tracks[0].boxes[i].cpu())
                    tracks[0].last_appear_boxes[i] = tracks[0].boxes[i]
            
            # If track is lost, add to gallery before marking as -1
            if tracks[0].disappear_time[i] == self.miss_tolerance and obj_id != -1:
                if self.use_reid and image is not None:
                    # Extract last known appearance
                    last_box = box_cxcywh_to_xyxy(tracks[0].boxes[i:i+1])
                    feat = self.reid_extractor.extract(image, last_box)
                    self.lost_gallery[obj_id] = {
                        'features': [feat],
                        'last_box': tracks[0].boxes[i].cpu(),
                        'time': 0
                    }
                tracks[0].ids[i] = -1

        # Clean up lost gallery
        to_remove = []
        for lid in self.lost_gallery:
            self.lost_gallery[lid]['time'] += 1
            if self.lost_gallery[lid]['time'] > self.max_lost_time:
                to_remove.append(lid)
        for lid in to_remove:
            del self.lost_gallery[lid]

        # Add newborn targets.
        new_tracks = TrackInstances(hidden_dim=tracks[0].hidden_dim,
                                    num_classes=tracks[0].num_classes,
                                    state_dim=tracks[0].state_dim,
                                    expand=tracks[0].expand,
                                    num_layers=tracks[0].num_layers,
                                    conv_dim=tracks[0].conv_dim)
        new_tracks_idxes = torch.max(model_outputs["scores"][0][:n_dets], dim=-1).values >= self.det_score_thresh
        new_tracks.logits = model_outputs["pred_logits"][0][:n_dets][new_tracks_idxes]
        new_tracks.boxes = model_outputs["pred_bboxes"][0][:n_dets][new_tracks_idxes]
        new_tracks.ref_pts = model_outputs["last_ref_pts"][0][:n_dets][new_tracks_idxes]
        new_tracks.scores = model_outputs["scores"][0][:n_dets][new_tracks_idxes]
        new_tracks.output_embed = model_outputs["outputs"][0][:n_dets][new_tracks_idxes]
        
        if self.use_dab:
            new_tracks.query_embed = model_outputs["aux_outputs"][-1]["queries"][0][:n_dets][new_tracks_idxes]
        else:
            new_tracks.query_embed = torch.cat(
                (
                    model_outputs["det_query_embed"][new_tracks_idxes][:, :256],    # hack
                    model_outputs["aux_outputs"][-1]["queries"][0][:n_dets][new_tracks_idxes]
                ),
                dim=-1
            )
        new_tracks.disappear_time = torch.zeros((len(new_tracks.logits), ), dtype=torch.long)
        new_tracks.labels = torch.max(new_tracks.scores, dim=-1).indices

        if self.use_motion:
            new_tracks.last_appear_boxes = model_outputs["pred_bboxes"][0][:n_dets][new_tracks_idxes]
        
        # Re-identification for newborn targets
        ids = []
        if self.use_reid and len(new_tracks) > 0 and len(self.lost_gallery) > 0 and image is not None:
            new_boxes = box_cxcywh_to_xyxy(new_tracks.boxes)
            new_features = self.reid_extractor.extract(image, new_boxes)
            
            # Match with lost gallery
            for i in range(len(new_tracks)):
                best_id = -1
                max_sim = -1
                
                for lid, data in self.lost_gallery.items():
                    # Compute max similarity with stored features
                    sim = torch.max(torch.stack([torch.mm(new_features[i:i+1], f.T) for f in data['features']]))
                    if sim > max_sim:
                        max_sim = sim
                        best_id = lid
                
                if max_sim > self.reid_thresh:
                    ids.append(best_id)
                    # Remove from gallery as it's now re-identified
                    del self.lost_gallery[best_id]
                else:
                    ids.append(self.max_obj_id)
                    self.max_obj_id += 1
        else:
            for i in range(len(new_tracks)):
                ids.append(self.max_obj_id)
                self.max_obj_id += 1
                
        new_tracks.ids = torch.as_tensor(ids, dtype=torch.long)
        new_tracks.hidden_state = torch.zeros((len(new_tracks.logits), tracks[0].num_layers, tracks[0].hidden_dim * tracks[0].expand, tracks[0].state_dim), dtype=torch.float)
        new_tracks.conv_history = torch.zeros((len(new_tracks.logits), tracks[0].num_layers, tracks[0].conv_dim, tracks[0].hidden_dim * tracks[0].expand), dtype=torch.float)
        new_tracks = new_tracks.to(new_tracks.logits.device)
        for _ in range(len(new_tracks)):
            obj_id = new_tracks.ids[_].item()
            self.motions[obj_id] = Motion(
                min_record_length=self.motion_min_length,
                max_record_length=self.motion_max_length
            )
            self.motions[obj_id].add_box(new_tracks.boxes[_].cpu())

        if self.visualize:
            visualize_ids += ids
            torch.save(torch.as_tensor(visualize_ids),
                       "./outputs/visualize_tmp/runtime_tracker/ids.tensor")

        return tracks, [new_tracks]

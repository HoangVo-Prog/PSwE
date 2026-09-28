"""Air-D scanpath sampling with the reference action/duration semantics."""

from __future__ import annotations

import numpy as np
import torch


class Sampling:
    def __init__(self, convLSTM_length=16, min_length=2, map_width=40, map_height=30, width=320, height=240):
        self.convLSTM_length = convLSTM_length
        self.min_length = min_length
        self.map_width = map_width
        self.map_height = map_height
        self.width = width
        self.height = height
        self.x_granularity = float(width / map_width)
        self.y_granularity = float(height / map_height)

    def random_sample(self, all_actions_prob, log_normal_mu, log_normal_sigma2):
        probs = all_actions_prob.detach().clone()
        probs[:, : self.min_length, 0] = 0
        actions = torch.distributions.Categorical(probs=probs).sample()
        selected_probs = torch.gather(all_actions_prob, 2, actions.unsqueeze(-1)).squeeze(-1)
        durations = torch.exp(torch.randn_like(log_normal_sigma2) * log_normal_sigma2 + log_normal_mu)
        return {
            "scanpath_length": self._lengths(actions),
            "durations": durations,
            "selected_actions_probs": selected_probs,
            "selected_actions": actions,
        }

    def _lengths(self, actions):
        lengths = actions.new_full((actions.shape[0],), self.convLSTM_length)
        for index in range(self.convLSTM_length):
            mask = (lengths == self.convLSTM_length) & (actions[:, index] == 0)
            lengths[mask] = index
        return lengths.unsqueeze(-1)

    def generate_scanpath(self, images, prob_sample_actions, durations, sample_actions):
        action_masks = images.new_zeros(prob_sample_actions.shape)
        duration_masks = images.new_zeros(prob_sample_actions.shape)
        predictions = []
        for index in range(images.shape[0]):
            actions = sample_actions[index].detach().cpu().numpy()
            times = durations[index].detach().cpu().numpy()
            fixations = []
            for order, action in enumerate(actions):
                if action == 0:
                    action_masks[index, order] = 1
                    break
                cell = int(action) - 1
                x = (cell % self.map_width) * self.x_granularity + self.x_granularity / 2
                y = (cell // self.map_width) * self.y_granularity + self.y_granularity / 2
                action_masks[index, order] = 1
                duration_masks[index, order] = 1
                fixations.append((x, y, times[order]))
            predictions.append(np.asarray(fixations, dtype={"names": ("start_x", "start_y", "duration"), "formats": ("f8", "f8", "f8")}))
        return predictions, action_masks, duration_masks


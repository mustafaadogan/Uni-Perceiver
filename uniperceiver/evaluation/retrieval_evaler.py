import os
import sys
import numpy as np
import torch
from uniperceiver.config import configurable
from .build import EVALUATION_REGISTRY
import json

@EVALUATION_REGISTRY.register()
class RetrievalEvaler(object):
    def __init__(self, cfg, annfile, output_dir,):
        super(RetrievalEvaler, self).__init__()
        pass

    def eval(self, vfeats, tfeats, index):

        similarity_scores_dict = {}

        for text_index in range(tfeats.size()[0]):
            relevant_video_features = vfeats[0]
            text_feature = tfeats[text_index]

            with torch.no_grad():
                score = (text_feature * relevant_video_features).sum(dim=-1).cpu().numpy()

            if index in similarity_scores_dict.keys():
                    similarity_scores_dict[index] += float(score)
            else:
                similarity_scores_dict[index] = float(score)

        return similarity_scores_dict[index]

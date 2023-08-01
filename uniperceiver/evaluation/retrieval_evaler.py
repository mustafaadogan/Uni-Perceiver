
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
        self.eval_bs = cfg.INFERENCE.EVAL_BS
        self.anno_file = cfg.INFERENCE.TEST_ANNFILE
        self.output_dir = output_dir
        pass

    def eval(self, vfeats, tfeats, labels, prefix=None):
        count = 0

        batch_size = self.eval_bs
        batch_num = vfeats.size(0) // batch_size
        batch_num_t = tfeats.size(0) // batch_num

        scores = []

        for i in range(batch_num):
            if i == batch_num - 1:
                b_tfeats = tfeats[i*batch_num_t:]
                b_vfeats = vfeats[i*batch_size:]
            else:
                b_tfeats = tfeats[i*batch_num_t:(i+1)*batch_num_t]
                b_vfeats = vfeats[i*batch_size:(i+1)*batch_size]

            with torch.no_grad():
                score = (b_tfeats.unsqueeze(1) * b_vfeats.unsqueeze(0)).sum(dim=-1).sum(dim=0).cpu().numpy()
                scores.append(float(score[0]))
                print(score)
        
        annoinfo = json.load(open(self.anno_file))
        main_task_results = {}
        prof_results = {}
        consumed_idx = 0

        for key in annoinfo.keys():
            main_task_results[key] = {"scores" : scores[consumed_idx : consumed_idx + 1 + len(annoinfo[key]["foils"])]}
            consumed_idx = 1 + len(annoinfo[key]["foils"])
            prof_results[key] = {"scores" : scores[consumed_idx : consumed_idx + 1 + len(annoinfo[key]["proficiency"]["foils"])]}
            consumed_idx = 1 + len(annoinfo[key]["proficiency"]["foils"])
        
        with open(f"{self.output_dir}/Main_Task_Results.json", "w") as task_outfile:
            json.dump(main_task_results, task_outfile, indent=4)

        with open(f"{self.output_dir}/Prof_Results.json", "w") as prof_outfile:
            json.dump(prof_results, prof_outfile, indent=4)

import torch
import numpy as np

def crop_membrane(membrane,low_lim=0.05,up_lim=0.95):
    for membrane_type in membrane.keys():
        q = np.array([[torch.quantile(membrane[membrane_type]['position'][:,_],low_lim),
               torch.quantile(membrane[membrane_type]['position'][:,_],up_lim)] for _ in range(3)])

        for _ in range(3):
            membrane[membrane_type]['distance'] = membrane[membrane_type]['distance'][
                torch.logical_and(membrane[membrane_type]['position'][:, _] > q[_][0],
                                  membrane[membrane_type]['position'][:, _] > q[_][0])]

            membrane[membrane_type]['position'] = membrane[membrane_type]['position'][torch.logical_and(membrane[membrane_type]['position'][:,_]>q[_][0],
                                                                                            membrane[membrane_type]['position'][:,_]>q[_][0])]


    return membrane
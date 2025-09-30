from pathlib import Path
from typing import Union
import numpy as np

import torch
import pandas as pd

import ot as pot


@torch.no_grad()
def compute_emd(p1, p2, device='cpu'):
    a_weights = torch.ones((p1.shape[0],), device=device) / p1.shape[0]
    b_weights = torch.ones((p2.shape[0],), device=device) / p2.shape[0]

    M = pot.dist(p1, p2).sqrt()
    return pot.emd2(a_weights, b_weights, M, numItermax=1e7)


def compute_window_avg(array, window_size):
    return np.convolve(array, np.ones(window_size), mode='valid') / window_size


def finish_results_table(data, timesteps):
    table = pd.DataFrame(
        {'time': [f"{t=}" for t in timesteps] + ["means"]}
    )
    data = pd.DataFrame(data)
    
    mean_row = [{
        name: val for name, val 
        in zip(data.columns, data.mean(axis=0))
    }]
    data = pd.concat(
        [data, pd.DataFrame(mean_row)], 
        ignore_index=True
    )
    table = pd.concat(
        [table, pd.DataFrame(data)], axis=1
    )
    
    results = [
        f"{round(avg, 3)}±{round(std, 3)}" for avg, std in 
        zip(table.iloc[:, 1:].mean(axis=1), table.iloc[:, 1:].std(axis=1))
    ]
    results[:-1] = [''] * (len(results) - 1)
    table["res"] = results

    table = table.round(3).set_index(table.columns[0])
    return table


def get_run_dir(run_id: str, search_dir: Union[str, Path] = './wandb'):
    found_dirs = [*Path(search_dir).glob(f"*{run_id}*")]
    if not found_dirs:
        raise FileNotFoundError(f"No directories found for run_id: {run_id}")
    return found_dirs[0] / 'files'

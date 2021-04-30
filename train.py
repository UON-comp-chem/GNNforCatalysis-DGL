import sys
import os
sys.path.append(os.getcwd())


from pymatgen.io.ase import AseAtomsAdaptor
from tqdm import tqdm

import os
import logging
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import ConcatDataset, Subset
from schnetpack.train import (CSVHook,EarlyStoppingHook, 
                          ReduceLROnPlateauHook, 
                          TensorboardHook)

from sklearn.metrics import mean_absolute_error
from catgnn.data_loader import batcher
from catgnn.trainer import Trainer
from catgnn.metrices import MeanAbsoluteError
from catgnn.mongo import make_atoms_from_doc
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from sklearn.model_selection import train_test_split
from pathlib import Path

# Initial parameters
MAIN = Path(os.getcwd())
TRIAL_NUM = 1
DATA_DIR = Path(MAIN, "data_cache/ch3-cache")
os.mkdir("data_cache") # Make cache dir to save data cache
device = torch.device("cuda")
MODEL_DIR = "./model-ch3-{}".format(TRIAL_NUM)

import pickle
with open(Path("./data/ch3-example.pkl"), 'rb') as fp:
    datasets = pickle.load(fp)
    
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(TRIAL_NUM)




### Now do the calculation ####
structures = [AseAtomsAdaptor.get_structure(make_atoms_from_doc\
                (doc['slab_information']['initial_configuration'])) \
                  for doc in datasets]
ls = [doc['lstags'] for doc in datasets]
y = [doc["energy"] for doc in datasets]

from catgnn.data_loader import CrystalGraphDatasetLS
dataset = CrystalGraphDatasetLS(root = DATA_DIR,
                        structures = structures,
                        labelsites = ls, 
                        targets = y, 
                        cutoff=4.5)
train_ratio = 0.675
val_ratio = 0.075
test_ratio = 0.25


trainval_idx,  test_idx= train_test_split(np.arange(dataset.__len__()), 
                            test_size = test_ratio, random_state = TRIAL_NUM)
train_idx, val_idx = train_test_split(trainval_idx, 
                                      test_size = val_ratio/(train_ratio+val_ratio),
                                      random_state = TRIAL_NUM)
train_dataset = Subset(dataset, train_idx)
val_dataset  = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)
train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=96, 
                            collate_fn=batcher(), 
                            shuffle=True, 
                            num_workers=4)
val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=96, 
                            collate_fn=batcher(), 
                            shuffle=False, 
                            num_workers=4)
test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=96, 
                            collate_fn=batcher(), 
                            shuffle=False, 
                            num_workers=4)

from catgnn.nmpeu import NMPEUModel
model = NMPEUModel(embed = "gpls",
                   dim=256,
                   cutoff=4.5,
                   output_dim=1,
                   num_gaussians = 96, 
                   n_conv=4,
                   act = "ssp",
                   aggregation_mode="avg",
                   norm=True)

model.set_mean_std(dataset.mean, dataset.std)

# build optimizer
optimizer = Adam(model.parameters(), lr=4.0e-4)

# hooks
logging.info("build trainer")
metrics = [MeanAbsoluteError("energy", model_output = None)]
hooks = [CSVHook(log_path=MODEL_DIR, metrics=metrics), 
         ReduceLROnPlateauHook(optimizer, factor = 0.75), 
         TensorboardHook(log_path=MODEL_DIR, metrics=metrics),
         EarlyStoppingHook(80)]
# trainer
loss = nn.MSELoss()
trainer = Trainer(
          MODEL_DIR,
          model=model,
          hooks=hooks,
          loss_fn=loss,
          optimizer=optimizer,
          train_loader=train_loader,
          validation_loader=val_loader,
)

# run training
logging.info("training")
print("Let's use", torch.cuda.device_count(), "GPUs!")
trainer.train(device=torch.device('cuda:0'))

# run testing
logging.info("test")
best_model = torch.load(trainer.best_model)
eval_args= {"device": torch.device('cuda:0')}
def run_prediction(args, model, data_loader):
    r"""Function used to run a prediction"""
    model.eval()
    pred = []
    y_true = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            bg, labels = batch_data.graph, batch_data.label
            labels = labels.to(args['device']) #, masks.to(args['device'])
            bg = bg.to(args['device'])
            prediction = model(bg)
            pred += prediction.cpu().numpy().tolist()
            y_true += labels.cpu().numpy().tolist()
    return np.array(pred), np.array(y_true)

y_pred, y_true = run_prediction(eval_args, best_model, test_loader)
mae = np.mean(np.abs(y_pred - y_true))
print("The {}th running, get MAE of {} eV".format(TRIAL_NUM, mae))




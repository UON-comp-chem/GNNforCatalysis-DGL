

"""https://github.com/kaist-amsg/LS-CGCNN-ens/blob/master/rawdata/MakeData.py"""
import sys
sys.path.append("/users/xinyu/github/GNNforCatalysis-DGL")

SYMBOL=sys.argv[1]
ADSAN = int(sys.argv[2])
ADSORBATE = {1:"hydrogen", 
             6:"carbon",
             7:"nitrogen",
             8:"oxygen",
             16:"sulfur"}[ADSAN]
import pickle
from pathlib import Path
with open(Path("../data/ch3-example.pkl"), 'rb') as fp:
    data = pickle.load(fp)
data = [doc for doc in data if 'initial_configuration' in doc.keys()]

from .util import CovalentRadius, pbcs
from catgnn.mongo import make_doc_from_atoms, make_atoms_from_doc
from tqdm import tqdm
from collections import defaultdict
import numpy as np
from ase.data import chemical_symbols
import pickle
import json
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_matcher import StructureMatcher
import os 
import csv
from ase.io import write
from ase.visualize import view
from ase.geometry import find_mic
from copy import deepcopy


    
def get_lalelled_site(atoms, site_type):
    assert site_type in ['A', 'A_A_A|FCC', 'A_A_A|HCP', 'A_A_A_A', 'A_A_B_B', 'A_A_B|FCC', 'A_A_B|HCP',
                       'A_A|A', 'A_A|B', 'A_B_B|FCC', 'A_B_B|HCP', 'A_B|A', 'A_B|B', 'B',
                       'B_B|A', 'B_B|B', 'subsurface', 'B_B_B|FCC', 'B_B_B|HCP'], \
                       "sorry, the adsortion site {} cannot be handelled.".format(site_type)
    pos = atoms.positions
    dist = find_mic(pos[12] - pos[:12, :], atoms.cell.array)[1]
    upperbound = 1.25
    if site_type in ["subsurface"]:
        criteria =[(CovalentRadius[chemical_symbols[int(n)]] +CovalentRadius[ADS_SYM]) \
               for n in atmos.get_atomic_numbers()]
        criteria = np.array(criteria)*upperbound
        PrimaryBindingSite = np.where(dists<criteria)[0]
        return PrimaryBindingSite, None

    if site_type in ['A', 'B']:
        site_type += "|FCC"
    elif site_type in ["A_A_B_B", 'A_A_A_A']:
        site_type += "|HCP" # This is difficult case, not standard, and lots of exceptions
#     print(site_type)
    psite, ssite = site_type.split("|")
    if len(psite) == 1:
        PrimaryBindingSite= np.argsort(dist)[:1]
    elif len(psite) == 3:
        PrimaryBindingSite= np.argsort(dist)[:2]
    elif len(psite)  == 5:
        PrimaryBindingSite= np.argsort(dist)[:3]
    elif len(psite)  == 7:
        PrimaryBindingSite= np.argsort(dist)[:4]
    
    dist[:4] += 1e9
    dist[-4:] += 1e9
    if len(ssite) == 1 or ssite == "HCP":
        SecondarySite = np.argsort(dist)[:1]
    else:
        SecondarySite = np.argsort(dist)[:3]
    return PrimaryBindingSite, SecondarySite
    


from pymatgen.io.ase import AseAtomsAdaptor
from ase import Atoms
from pymatgen.analysis.local_env import VoronoiNN
ADS_SYM = chemical_symbols[ADSAN]
newdata = []
upperbound = 1.25
mat = StructureMatcher(primitive_cell=False)
bound = {"H" :[-3.0, 3.0],
         "C":[-1.0, 9.0],
         "N":[-4., 6.],
         "O":[-4.5, 4.5],
         "S":[-4.5, 2.]}[ADS_SYM]


for _idx, datum in tqdm(enumerate(data)):
    datum['adsorption_energy'] = datum['energy']
    if datum['adsorption_energy'] < bound[0] or datum['adsorption_energy'] > bound[1]:
        print("discard {} because of energy filter".format(_idx))
        continue
    ## Find binding site atoms
    ### Get atoms
    atmsi = make_atoms_from_doc(datum['initial_configuration'])
    atms  = make_atoms_from_doc(datum)
    ### Adsorbate index
    PrimaryBinding, SecondaryBinding = get_lalelled_site(atms, datum['site_type'])
    pos = atms.positions
    dists = find_mic(pos[12] - pos[:12, :], atms.cell.array)[1]
    assert atmsi.get_atomic_numbers()[12] == ADSAN
    assert atms.get_atomic_numbers()[12] == ADSAN
    ### apply criteria
    if len(PrimaryBinding) == 0:
        print("discard {} because of no binding site found".format(_idx))
        continue # no binding site.
    
    ## Apply Labeling
    ### Relaxed surface Input
    initsurf = atmsi.copy()[:12]
    tags = initsurf.get_tags()
    tags[PrimaryBinding] = 1
    if SecondaryBinding is not None:
        tags[SecondaryBinding] = 2
    initsurf.set_tags(tags)
    dists = dists
    

    datum['slab_information'] = {}
    newinfo = make_doc_from_atoms(initsurf)
    
    datum['slab_information']['initial_configuration'] = newinfo
    ## record data
    ### Process neighbors
    ### Channel
    datum['lstags'] = tags.tolist()
    newdata.append(datum)

with open("./data/ch3-labelled.pkl".format(SYMBOL.lower()), 'wb') as fp:
    pickle.dump(newdata, fp)


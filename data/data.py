"""
    File to load dataset based on user control from main file
"""
from JOBSynthetic import JOBSyntheticDataset
from JOBSyntheticBig import JOBSyntheticBigDataset
from JCCHTPCH import JCCHTPCHDataset
from JOBScale import JOBScaleDataset



def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file
        returns:
        ; dataset object
    """
    if DATASET_NAME == "Logical-Scale" or DATASET_NAME == "Compact-Scale" or DATASET_NAME == "Query-Plan-Oriented-Scale" or DATASET_NAME == "Fine-Grained-Scale":
        return JOBScaleDataset(DATASET_NAME)

    if DATASET_NAME == "compact-jcch":
        return JCCHTPCHDataset(DATASET_NAME)

    if DATASET_NAME == 'Compact-Big' or DATASET_NAME == 'Logical-Big' or DATASET_NAME == 'Logical-Big-Samples' or DATASET_NAME == 'Compact-Big-Samples':
        return JOBSyntheticBigDataset(DATASET_NAME)

    if DATASET_NAME == 'Compact' or DATASET_NAME == 'Compact-Samples' or DATASET_NAME == 'Logical-Samples' or DATASET_NAME == 'Fine-Grained' or DATASET_NAME == 'Logical' or DATASET_NAME == 'Physical' or DATASET_NAME == 'Query-Plan-Oriented':
        return JOBSyntheticDataset(DATASET_NAME)

    # handling for MNIST or CIFAR Superpixels
    if DATASET_NAME == 'MNIST' or DATASET_NAME == 'CIFAR10':
        return SuperPixDataset(DATASET_NAME)
    
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'ZINC' or DATASET_NAME == 'ZINC-full':
        return MoleculeDataset(DATASET_NAME)

    # handling for the TU Datasets
    TU_DATASETS = ['ENZYMES', 'DD', 'PROTEINS_full']
    if DATASET_NAME in TU_DATASETS: 
        return TUsDataset(DATASET_NAME)

    # handling for SBM datasets
    SBM_DATASETS = ['SBM_CLUSTER', 'SBM_PATTERN']
    if DATASET_NAME in SBM_DATASETS: 
        return SBMsDataset(DATASET_NAME)
    
    # handling for TSP dataset
    if DATASET_NAME == 'TSP':
        return TSPDataset(DATASET_NAME)

    # handling for COLLAB dataset
    if DATASET_NAME == 'OGBL-COLLAB':
        return COLLABDataset(DATASET_NAME)

    # handling for the CSL (Circular Skip Links) Dataset
    if DATASET_NAME == 'CSL': 
        return CSLDataset(DATASET_NAME)
    
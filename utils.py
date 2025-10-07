from rdkit.Chem import Descriptors

def getMolDescriptors(mol, missingVal=None):
    """Calculate the full list of descriptors for a RDKit molecule.
    
    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule object for which to calculate descriptors.
    missingVal : Any, optional
        Value to use if a descriptor cannot be calculated. Default is None.
        
    Returns
    -------
    dict
        Dictionary containing descriptor names as keys and calculated values as values.
    """
    res = {}
    for nm,fn in Descriptors._descList:
        # some of the descriptor fucntions can throw errors if they fail, catch those here:
        try:
            val = fn(mol)
        except:
            import traceback
            traceback.print_exc()
            # and set the descriptor value to whatever missingVal is
            val = missingVal
        res[nm] = val
    return res
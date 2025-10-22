from rdkit.Chem import Draw
from IPython.display import display

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import networkx as nx
from rdkit.Chem import PandasTools, AllChem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import SimilarityMaps

import numpy as np



def extract_similar_substructures(mol, similarity_map, threshold=0.2):
    """
    Extract substructures from a molecule where connected atoms have similar similarity values.
    
    Parameters:
    - mol: RDKit molecule object
    - similarity_map: dict mapping atom indices to similarity values
    - threshold: maximum difference in similarity values for atoms to be grouped together
    
    Returns:
    - list of tuples: (atom_indices, mean_similarity, substructure_mol)
    """
    
    # Create a graph representation of the molecule
    G = nx.Graph()
    
    # Add atoms as nodes with their similarity values
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        G.add_node(idx, similarity=similarity_map.get(idx, 0.0))
    
    # Add bonds as edges
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        G.add_edge(begin_idx, end_idx)
    
    # Find connected components with similar similarity values
    visited = set()
    substructures = []
    
    def dfs_similar_atoms(start_atom, current_group, target_similarity):
        """DFS to find connected atoms with similar similarity values"""
        if start_atom in visited:
            return
        
        visited.add(start_atom)
        atom_similarity = G.nodes[start_atom]['similarity']
        
        # Check if this atom's similarity is within threshold of target
        if abs(atom_similarity - target_similarity) <= threshold:
            current_group.append(start_atom)
            
            # Explore neighbors
            for neighbor in G.neighbors(start_atom):
                if neighbor not in visited:
                    dfs_similar_atoms(neighbor, current_group, target_similarity)
    
    # Process each unvisited atom
    for atom_idx in range(mol.GetNumAtoms()):
        if atom_idx not in visited:
            group = []
            target_sim = G.nodes[atom_idx]['similarity']
            dfs_similar_atoms(atom_idx, group, target_sim)
            
            if group:  # Only add non-empty groups
                mean_similarity = np.mean([G.nodes[idx]['similarity'] for idx in group])
                
                # Create substructure molecule if possible
                try:
                    submol = Chem.PathToSubmol(mol, group)
                    substructures.append((group, mean_similarity, submol))
                except:
                    # If PathToSubmol fails, still include the group without submol
                    substructures.append((group, mean_similarity, None))
    
    # Sort by mean similarity (descending)
    substructures.sort(key=lambda x: x[1], reverse=True)
    
    return substructures

def get_similarity_map_from_model(mol, model):
    """
    Generate similarity map values for each atom in the molecule.
    
    Parameters:
    - mol: RDKit molecule object
    - model: trained model with predict_proba method
    
    Returns:
    - dict: mapping atom indices to similarity values
    """
    # Get baseline prediction for the full molecule
    full_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    baseline_pred = model.predict_proba(np.array([full_fp]))[0][1]
    
    similarity_map = {}
    
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        # Get fingerprint with this atom removed/modified
        fp = SimilarityMaps.GetMorganFingerprint(mol, idx)
        pred_val = model.predict_proba(np.array([fp]))[0][1]
        # Calculate the difference (contribution of this atom)
        delta = baseline_pred - pred_val
        similarity_map[idx] = delta
    
    return similarity_map

def visualize_substructures(mol, substructures, top_n=5):
    """
    Visualize the top N substructures with their similarity values.
    
    Parameters:
    - mol: RDKit molecule object
    - substructures: list of (atom_indices, mean_similarity, submol) tuples
    - top_n: number of top substructures to display
    """
    
    print(f"Top {top_n} substructures by similarity:")
    print("-" * 50)
    
    for i, (atom_indices, mean_sim, submol) in enumerate(substructures[:top_n]):
        print(f"\nSubstructure {i+1}:")
        print(f"Atom indices: {atom_indices}")
        print(f"Mean similarity: {mean_sim:.4f}")
        print(f"Number of atoms: {len(atom_indices)}")
        
        # Highlight atoms in the original molecule
        if len(atom_indices) > 1:  # Only highlight if more than one atom
            img = Draw.MolToImage(mol, highlightAtoms=atom_indices, size=(300, 300))
            display(img)

def topology_from_rdkit(rdkit_molecule):

    topology = nx.Graph()
    for atom in rdkit_molecule.GetAtoms():
        # Add the atoms as nodes
        topology.add_node(atom.GetIdx())

        # Add the bonds as edges
        for bonded in atom.GetNeighbors():
            topology.add_edge(atom.GetIdx(), bonded.GetIdx())

    return topology


def is_isomorphic(topology1, topology2):
    return nx.is_isomorphic(topology1, topology2)

def extract_substructures_with_graph_matching(data, model, sample_size=100, threshold=0.05, random_state=42):
    """
    Extract substructures and group them by exact molecular graph isomorphism.
    Returns:
    - dict: {substructure_index: {'mol': mol_object, 'similarities': [values]}}
    """
    np.random.seed(random_state)
    sample_indices = np.random.choice(len(data), min(sample_size, len(data)), replace=False)
    all_substructures = []

    print(f"Processing {len(sample_indices)} molecules...")

    for idx in sample_indices:
        mol = data.iloc[idx]['molecule']
        if mol is None:
            continue
        try:
            similarity_map = get_similarity_map_from_model(mol, model)
            substructures = extract_similar_substructures(mol, similarity_map, threshold)
            for atom_indices, mean_sim, submol in substructures:
                if submol is not None and len(atom_indices) >= 2:
                    try:
                        topology = topology_from_rdkit(submol)
                        all_substructures.append({
                            'mol': submol,
                            'topology': topology,
                            'similarity': float(mean_sim),
                            'atom_count': len(atom_indices)
                        })
                    except:
                        continue
        except Exception as e:
            continue

    print(f"Found {len(all_substructures)} total substructures")
    grouped_substructures = []
    used = [False] * len(all_substructures)

    print("Grouping identical substructures by graph isomorphism...")

    for i, sub_i in enumerate(all_substructures):
        if used[i]:
            continue
        group = [sub_i]
        used[i] = True
        for j in range(i+1, len(all_substructures)):
            if used[j]:
                continue
            sub_j = all_substructures[j]
            if is_isomorphic(sub_i['topology'], sub_j['topology']):
                group.append(sub_j)
                used[j] = True
        grouped_substructures.append(group)

    # Convert to result format and filter by minimum occurrences
    result = {}
    group_id = 0
    for group in grouped_substructures:
        if len(group) >= 2:
            similarities = [s['similarity'] for s in group]
            result[group_id] = {
                'mol': group[0]['mol'],
                'similarities': similarities,
                'count': len(similarities)
            }
            group_id += 1

    return result

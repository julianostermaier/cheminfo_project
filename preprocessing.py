

import re
from typing import Optional, Sequence, Union, List
import math

import pandas as pd
import numpy as np


def _standardize_to_nM(value: float, unit: Optional[str], assume_unit: Optional[str] = None) -> Optional[float]:
    """Convert a numeric binding value and unit to nM.

    Returns float in nM, or None if conversion not possible.
    """
    try:
        val = float(value)
    except Exception:
        return None
    if unit is None or unit == '':
        if assume_unit:
            unit = assume_unit
        else:
            return None

    u = unit.lower().strip()
    # normalize common unit spellings
    u = u.replace('\u00b5', 'u')  # micro sign -> u
    # allow units like 'nm', 'nM', 'nanomolar', 'uM', 'um', 'micromolar', etc.
    if u in ('nm', 'nanomolar', 'nanomoles', 'nanomole'):
        return val
    if u in ('pm', 'picomolar', 'picomoles', 'picomole'):
        return val / 1000.0
    if u in ('um', 'umolar', 'umol', 'micromolar', 'micromoles', 'micromole', 'uM'):
        return val * 1000.0
    if u in ('mm', 'mmolar', 'mmol', 'millimolar', 'millimoles', 'millimole'):
        return val * 1_000_000.0
    if u in ('m', 'molar', 'mol'):
        return val * 1_000_000_000.0
    # if unit is just single character like 'n' or 'p'
    if u == 'n':
        return val
    if u == 'p':
        return val / 1000.0

    # try to match common patterns
    if re.match(r'^p', u):
        return val / 1000.0
    if re.match(r'^n', u):
        return val
    if re.match(r'^(u|micro)', u):
        return val * 1000.0
    if re.match(r'^m', u):
        return val * 1_000_000.0

    return None



def _extract_value_and_unit(s: str, variable: str, allow_fallback: bool = True) -> Optional[tuple]:
    """Extract numeric value and unit for a given variable (Kd/Ki) from text string s.

    Returns (value_str, unit_str) or None.
    """
    if not isinstance(s, str):
        return None
    # Look for patterns like 'Kd=0.006uM', 'Kd ~ 500 uM', 'Ki: 1.2 nM', or 'Kd 5 nM'
    pattern = rf'(?i)\b{re.escape(variable)}\b\s*[=:~]?\s*([0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?)\s*([a-zA-Z\u00B5μ]+)?'
    m = re.search(pattern, s)
    if m:
        val = m.group(1)
        unit = m.group(3) or ''
        return val, unit

    # If variable label not present, optionally try to find any numeric with unit and assume it's the requested value
    if allow_fallback:
        m2 = re.search(r'([0-9]+\.?[0-9]*([eE][-+]?[0-9]+)?)\s*([a-zA-Z\u00B5μ]+)', s)
        if m2:
            return m2.group(1), (m2.group(3) or '')

    return None


def PDBpreprocessing(
    data: pd.DataFrame,
    category: Optional[str] = 'hydrolase',
    variable: Union[str, Sequence[str]] = 'Kd',
    keep_protein_chain: bool = False,
    keep_raw: bool = False,
    log_transform: bool = True,
    assume_unit: Optional[str] = None,
) -> pd.DataFrame:
    """Preprocess a PDBBind-like DataFrame.

    Parameters
    - data: pandas DataFrame containing at least SMILES and raw KD/KI text columns.
    - category: optional string to filter rows by presence in a 'header' column (case-insensitive).
      If None, no filtering by category is applied.
    - variable: 'Kd', 'Ki', 'both', or a list/sequence of variables to extract.
    - keep_protein_chain: when True, attempt to keep a protein chain/sequence column if present.
    - log_transform: when True and a single variable is requested, create a 'target' column as ln(value_in_nM).
    - assume_unit: if unit is missing in raw text, assume this unit (e.g., 'nM', 'uM'). If None, rows with no unit are dropped.

    Returns a cleaned DataFrame with standardized columns added:
    - for each requested variable X: a column 'X_nM' with float values in nM
    - if single variable and log_transform=True: a 'target' column containing ln(X_nM)

    Notes
    - The function is tolerant to a variety of text encodings like 'Kd=0.006uM', 'Kd~500uM', 'Ki: 5 nM'.
    - If the input DataFrame uses different column names, the function will try to detect common names
      for the binding text column and for SMILES. If SMILES can't be found the function will raise.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError('data must be a pandas DataFrame')

    df = data.copy()

    # Use exact expected column names from the dataset header.
    # This function is strict: it expects the input DataFrame to have the columns
    # exactly as in your example: 'smiles' and 'kd/ki'. If they are missing, raise.
    smiles_col = 'smiles'
    binding_col = 'kd/ki'
    seq_col = 'seq'

    if smiles_col not in df.columns:
        raise KeyError('Expected a column named "smiles" in the input DataFrame.')
    if binding_col not in df.columns:
        raise KeyError('Expected a column named "kd/ki" in the input DataFrame.')
    # seq is optional unless requested
    if keep_protein_chain and seq_col not in df.columns:
        raise KeyError('keep_protein_chain=True but expected a column named "seq" in the input DataFrame.')

    # filter by category if requested
    if category is not None:
        if 'header' in df.columns:
            df = df[df['header'].str.contains(category, case=False, na=False)].copy()
        else:
            # if no header column, do not filter but warn (silently continue)
            pass

    # prepare variable list
    if isinstance(variable, str):
        if variable.lower() == 'both':
            variables = ['Kd', 'Ki']
        else:
            variables = [variable]
    else:
        variables = list(variable)

    # Extract and standardize each requested variable
    # helper: detect which label (Kd or Ki) is present in the binding text
    def _detect_label(raw_text: str) -> Optional[str]:
        if not isinstance(raw_text, str):
            return None
        m = re.search(r'(?i)\b(Kd|Ki)\b', raw_text)
        if m:
            return m.group(1)
        return None
    for var in variables:
        column_name = f'{var}_nM'

        def _extract_row_value(row: pd.Series):
            raw_text = str(row[binding_col]) if binding_col in row.index else ''
            label = _detect_label(raw_text)
            # if a single variable requested, require the label to match exactly (case-insensitive)
            if len(variables) == 1:
                if label is None or label.lower() != var.lower():
                    return None

            # Prefer numeric 'value' column if available in the provided dataset (interpreted as pKd/pKi)
            # Only use it when the binding label in the text matches the requested variable (for single-variable calls).
            if 'value' in row.index and pd.notna(row['value']):
                try:
                    # If single variable requested, ensure the label matches
                    if len(variables) == 1 and label is not None and label.lower() != var.lower():
                        # label mismatch -> don't use this row for this variable
                        return None
                    pv = float(row['value'])
                    # value is pKd/pKi = -log10(Kd_M)
                    kd_m = 10.0 ** (-pv)
                    kd_nM = kd_m * 1e9
                    return float(kd_nM)
                except Exception:
                    pass

            # otherwise parse the 'kd/ki' text (e.g. 'Kd=0.006uM')
            # when only one variable requested, do NOT allow a generic numeric-with-unit fallback
            allow_fallback = not (len(variables) == 1)
            parsed = _extract_value_and_unit(raw_text, var, allow_fallback=allow_fallback)
            if parsed is None:
                return None
            val_str, unit_str = parsed
            return _standardize_to_nM(val_str, unit_str, assume_unit=assume_unit)

        df[column_name] = df.apply(_extract_row_value, axis=1)

    # drop rows missing SMILES or missing all requested variables
    required_cols = [f'{v}_nM' for v in variables]
    df = df[df[smiles_col].notna()].copy()
    # keep rows where all required columns (for single variable) or at least one (for both) are present
    if len(required_cols) == 1:
        df = df[df[required_cols[0]].notna()].copy()
    else:
        df = df[df[required_cols].notna().any(axis=1)].copy()

    # create target(s) as natural log of nM values
    target_cols: List[str] = []
    if len(variables) == 1:
        col = required_cols[0]
        if log_transform:
            df = df[df[col] > 0].copy()
            df['target'] = df[col].apply(lambda x: math.log(x))
        else:
            df['target'] = df[col]
        target_cols = ['target']
    else:
        # create separate named targets for each variable
        for v, col in zip(variables, required_cols):
            tgt = 'target'
            if log_transform:
                df = df[df[col].notna() & (df[col] > 0)].copy()
                df[tgt] = df[col].apply(lambda x: math.log(x))
            else:
                df[tgt] = df[col]
        target_cols.append(tgt)

    # prepare output columns: only keep smiles, the target(s), and seq if requested
    out_cols: List[str] = [smiles_col] + target_cols
    if keep_protein_chain:
        out_cols.append(seq_col)

    # final reduced DataFrame
    return df.loc[:, out_cols].reset_index(drop=True)


if __name__ == '__main__':
    # quick smoke test when running the module directly
    print('preprocessing module loaded. Use PDBpreprocessing(data, ...) to clean your DataFrame.')

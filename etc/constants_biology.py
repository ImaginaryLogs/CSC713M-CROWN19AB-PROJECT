from itertools import product
from etc import constants_labels
CONJOINT_TRIADS = {
    'A': '1', 'G': '1', 'V': '1',             # Aliphatic
    'I': '2', 'L': '2', 'F': '2', 'P': '2',   # Large Aliphatic
    'Y': '3', 'M': '3', 'T': '3', 'S': '3',   # Polar
    'H': '4', 'N': '4', 'Q': '4', 'W': '4',   # Aromatic/Neutral
    'R': '5', 'K': '5',                       # Positive Charge
    'D': '6', 'E': '6',                       # Negative Charge
    'C': '7'                                  # Cysteine (Special)
}

# Kyte-Doolittle Hydrophobicity Scale
HYDRO_MAP = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 
             'G': -0.4, 'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 
             'P': -1.6, 'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}

# Charge at pH 7.4 (Approximation for pI intensity)
# https://ia601308.us.archive.org/27/items/CRC.Press.Handbook.of.Chemistry.and.Physics.85th.ed.eBook-LRN/CRC.Press.Handbook.of.Chemistry.and.Physics.85th.ed.eBook-LRN.pdf
CHARGE_MAP = {
    'A'   
}    


TRIAD_NAMES = ["".join(t) for t in product("1234567", repeat=3)]
KMER3_NAMES = ["".join(t) for t in product(constants_labels.AMINO_ACID_ALPHABETS, repeat=3)]
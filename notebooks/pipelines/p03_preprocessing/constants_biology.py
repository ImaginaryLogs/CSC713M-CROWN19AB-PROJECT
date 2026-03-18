from itertools import product
import constants_labels
CONJOINT_TRIADS = {
    'A': '1', 'G': '1', 'V': '1',             # Aliphatic
    'I': '2', 'L': '2', 'F': '2', 'P': '2',   # Large Aliphatic
    'Y': '3', 'M': '3', 'T': '3', 'S': '3',   # Polar
    'H': '4', 'N': '4', 'Q': '4', 'W': '4',   # Aromatic/Neutral
    'R': '5', 'K': '5',                       # Positive Charge
    'D': '6', 'E': '6',                       # Negative Charge
    'C': '7'                                  # Cysteine (Special)
}


TRIAD_NAMES = ["".join(t) for t in product("1234567", repeat=3)]
KMER3_NAMES = ["".join(t) for t in product(constants_labels.AMINO_ACID_ALPHABETS, repeat=3)]
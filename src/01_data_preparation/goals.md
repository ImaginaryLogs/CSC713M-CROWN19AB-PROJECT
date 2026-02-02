# Theoretical Framework

In order to prepare for the data exploration, we have to define the scope of the dataset.

Note, Here are the general features that is to explore:

First Pass :

- Protein Sequence Embedding via own k-mer motifs or thorugh ProtBERT
- Physichochemical Property Embedding via libraries such as Biopython.SeqUtils.ProtParam's ProteinAnalysis (Isoelectric Point, Instability Index, etc...)

Once finished, the Second Pass:

- Topological/Structural features (such as length)
- Interaction features based on taking the difference or product of the original viruses' properties with the Antibody (Difference in Isoelectric point of the Wuhan Strain and an Antibody).

This way, we can have an idea what we should be looking at.

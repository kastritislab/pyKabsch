# pyKabsch  
A numpy implementation of the Kabsch algorithm for fast alignment of protein structures.

A small implementation of the Kabsch algorithm to align many protein structures against a single reference.  
The functions operate entirely on NumPy arrays and avoid Python-level loops for efficiency.  
Both single-structure and batched alignments are supported, returning rotation matrices, translations, and RMSDs.  
The implementation enforces proper rotations and uses vectorized SVD for high throughput.

### Usage  
Input coordinates must be provided as arrays of shape `[L, 3]` for single models or `[N, L, 3]` for batches, where `L` is the number of atoms.  
Use `rmsd_kabsch_multi(anchor, mobiles)` to compute RMSDs and obtain the corresponding rotation and translation.  
To produce aligned coordinates, apply the transformation via `apply_transform(coords, R, t)`.

# Misc
Tested with: numpy==2.3.2

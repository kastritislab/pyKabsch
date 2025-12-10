import numpy as np

def rmsd_kabsch_multi(anchor, mobiles):
    """
    Kabsch-aligned RMSD between one anchor and one or many mobiles.

    Parameters
    ----------
    anchor : array_like, shape (L, 3)
        Reference coordinates.
    mobiles : array_like, shape (L, 3) or (N, L, 3)
        Mobile coordinates to be aligned onto `anchor`.

    Returns
    -------
    rmsd : float or ndarray, shape (N,)
        RMSD(s) after optimal superposition.
    R : ndarray, shape (3,3) or (N,3,3)
        Rotation matrix (or matrices) that aligns each mobile onto `anchor`.
    t : ndarray, shape (3,) or (N,3)
        Translation vector(s) such that:

            aligned = mobiles @ R + t

        gives coordinates aligned to `anchor`.

    Notes
    -----
    - Uses standard Kabsch: H = A^T B, SVD(H) = U S V^T, R = V U^T.
    - Proper rotation enforced (det(R) = +1).
    - For mobiles with shape (N,L,3), everything is done in batch.
    """
    A = np.asarray(anchor, dtype=np.float64)
    B = np.asarray(mobiles, dtype=np.float64)

    if A.ndim != 2 or A.shape[1] != 3:
        raise ValueError(f"anchor must have shape (L, 3), got {A.shape}")

    single = False
    if B.ndim == 2 and B.shape[1] == 3:
        # Promote single mobile to batch of size 1
        B = B[None, ...]
        single = True
    elif B.ndim != 3 or B.shape[2] != 3:
        raise ValueError(f"mobiles must have shape (L,3) or (N,L,3), got {B.shape}")

    L = A.shape[0]
    if B.shape[1] != L:
        raise ValueError(f"anchor and mobiles must have same L, got {L} vs {B.shape[1]}")

    # centroids
    centroid_A = A.mean(axis=0)          # (3,)
    centroid_B = B.mean(axis=1)          # (N,3)

    # center
    A0 = A - centroid_A                  # (L,3)
    B0 = B - centroid_B[:, None, :]      # (N,L,3)

    # covariance H_n = A0^T B0_n, shape (N,3,3)
    H = np.einsum('la,nlb->nab', A0, B0)

    # batched SVD
    U, S, Vt = np.linalg.svd(H)
    V  = np.swapaxes(Vt, -2, -1)         # (N,3,3)
    UT = np.swapaxes(U,  -2, -1)         # (N,3,3)

    # proper rotation: det(R) = +1
    R = V @ UT                           # (N,3,3)
    detR = np.linalg.det(R)              # (N,)
    neg = detR < 0.0
    if np.any(neg):
        V[neg, :, -1] *= -1.0
        R = V @ UT

    # rotate B0: (N,L,3)
    B_rot = np.einsum('nlb,nba->nla', B0, R)

    # aligned coordinates: B_rot + centroid_A
    A_expanded = centroid_A[None, None, :]        # (1,1,3)
    B_aligned = B_rot + A_expanded                # (N,L,3)

    # RMSDs per mobile
    diff = B_aligned - A[None, :, :]              # (N,L,3)
    rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=-1), axis=-1))  # (N,)

    # translation vectors t_n such that:
    # aligned = B @ R + t
    # From B_aligned = (B - cB) R + cA = B R + (cA - cB R) => t = cA - cB R
    cB_rot = np.einsum('na,nab->nb', centroid_B, R)   # (N,3)
    t = centroid_A[None, :] - cB_rot                  # (N,3)

    if single:
        return float(rmsd[0]), R[0], t[0]
    else:
        return rmsd, R, t

def apply_transform(coords, R, t):
    """
    Apply a rigid transform to coordinates.

    Parameters
    ----------
    coords : ndarray, shape (..., L, 3)
        Input coordinates.
    R : ndarray, shape (3,3)
        Rotation matrix.
    t : ndarray, shape (3,)
        Translation vector.

    Returns
    -------
    ndarray, shape (..., L, 3)
        Transformed coordinates: coords @ R + t
    """
    coords = np.asarray(coords, float)
    R = np.asarray(R, float)
    t = np.asarray(t, float)
    return coords @ R + t

def calc_rmsd(A, B):
    """
    Compute RMSD between two identically shaped ndarrays.

    Parameters
    ----------
    A, B : ndarray, shape (..., L, 3)

    Returns
    -------
    float or ndarray
        RMSD over last two axes.
        If shape is (L,3) -> scalar.
        If shape is (N,L,3) -> (N,)
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)

    diff = A - B                   # (..., L, 3)
    sq = np.sum(diff * diff, axis=-1)  # (..., L)
    return np.sqrt(np.mean(sq, axis=-1))

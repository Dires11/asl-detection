import numpy as np

# Fingertip and MCP/DIP landmark indices (MediaPipe hand model)
_TIPS = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
_MCPS = [1, 5,  9, 13, 17]   # corresponding MCP joints
_DIPS = [7, 11, 15, 19]      # DIP joints for index, middle, ring, pinky (no thumb DIP)


def _extra_features(pts_norm: np.ndarray) -> np.ndarray:
    """24 geometry features computed on the already-normalised (21,3) array.

    - 10 tip-to-tip distances: all pairs among the 5 fingertips.
      Encodes V-spread vs U-together, thumb-out (Y/A) vs curled.
    - 5 MCP→tip distances: one per finger.
      Encodes curl depth, helping distinguish A / S / E fist-like shapes.
    - 4 thumb-tip→DIP distances (index/middle/ring/pinky DIP joints).
      In M the thumb is tucked under 3 fingers' DIPs, N under 2, S crosses
      differently — the pattern of these 4 distances is the key discriminator.
    - 5 fingertip→wrist distances (wrist is origin, so just the tip norms).
      Tighter fist = shorter; subtle differences separate S / A / E / T.
    """
    dists = []
    # 10 tip-to-tip
    for i in range(len(_TIPS)):
        for j in range(i + 1, len(_TIPS)):
            dists.append(np.linalg.norm(pts_norm[_TIPS[i]] - pts_norm[_TIPS[j]]))
    # 5 MCP→tip (curl depth)
    for mcp, tip in zip(_MCPS, _TIPS):
        dists.append(np.linalg.norm(pts_norm[tip] - pts_norm[mcp]))
    # 4 thumb-tip→DIP
    for dip in _DIPS:
        dists.append(np.linalg.norm(pts_norm[4] - pts_norm[dip]))
    # 5 fingertip→wrist (wrist at origin)
    for tip in _TIPS:
        dists.append(np.linalg.norm(pts_norm[tip]))
    return np.array(dists)


def normalize_landmarks(landmarks) -> np.ndarray | None:
    """Return an 84-dim feature vector (position+scale invariant).

    Accepts MediaPipe landmark objects OR a numpy array of shape (21, 3).
    Returns None if the hand detection is unreliable (near-zero scale).

    Feature layout:
      [0:60]  — 20 normalised landmark coords (wrist dropped), flattened
      [60:84] — 24 geometry features (see _extra_features)
    """
    if hasattr(landmarks[0], 'x'):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    else:
        pts = np.array(landmarks).reshape(21, 3)
    wrist = pts[0]
    translated = pts - wrist                    # wrist → origin
    scale = np.linalg.norm(translated[5])       # wrist-to-index-MCP distance
    if scale < 1e-6:
        return None
    pts_norm = translated / scale               # shape (21, 3)
    coords   = pts_norm[1:].flatten()           # shape (60,)
    extra    = _extra_features(pts_norm)        # shape (24,)
    return np.concatenate([coords, extra])      # shape (84,)

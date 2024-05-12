import numpy as np

# read the flow file
def read_flow(path):
    with open(path, 'rb') as f:
        # read the header
        header = f.read(4)
        if header.decode('utf-8') != 'PIEH':
            print("Error: Flow file header does not contain PIEH")
        w = np.fromfile(f, np.int32, 1).squeeze()
        h = np.fromfile(f, np.int32, 1).squeeze()

        # read the data
        data = np.fromfile(f, np.float32)
        data = np.resize(data, (h, w, 2))
        # print(f"Flow file data shape: {data.shape}")

    return data


def make_colorwheel():
    """
    Creates a color wheel for flow visualization.

    Returns:
    A NumPy array of shape (ncols, 3) representing the color wheel.
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros((ncols, 3))  # RGB channels

    col = 0
    # RY
    colorwheel[:RY, 0] = 255
    colorwheel[:RY, 1] = np.floor(255 * (np.arange(RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255 * (np.arange(YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255 * (np.arange(GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255 * (np.arange(CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255 * (np.arange(BM) / BM))
    col += BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255 * (np.arange(MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def compute_color(u, v):
    """
    Converts normalized flow vectors to a color image.

    Args:
        u: A NumPy array of the same shape as v representing the horizontal 
            flow component (normalized between -1 and 1).
        v: A NumPy array of the same shape as u representing the vertical 
            flow component (normalized between -1 and 1).

    Returns:
        A NumPy array of shape (height, width, 3) representing the color-coded
        flow image.
    """

    # Handle NaN values
    nan_mask = (np.isnan(u) | np.isnan(v))
    u[nan_mask] = 0
    v[nan_mask] = 0

    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]

    # Calculate magnitude and angle
    rad = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi

    # Normalize angle to 0 to ncols-1 range
    fk = ((a + 1) / 2) * (ncols - 1)

    # Map angles to color wheel indices
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    img = np.zeros((u.shape[0], u.shape[1], 3))

    # Loop through color channels
    for i in range(3):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        # Increase saturation with radius
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])

        # Handle out-of-range values
        col[~idx] *= 0.75

        # Apply mask for NaN values
        img[:, :, i] = np.array(np.floor(255 * col) * (1 - nan_mask)).astype(np.uint8)
        # img[nan_mask] = 0

    return img

def colorToFlow(img):
    colorwheel = make_colorwheel()
    ncols = colorwheel.shape[0]
    
    height, width, _ = img.shape
    flow = np.zeros((height, width, 2))
    
    for i in range(height):
        for j in range(width):
            col = np.array([img[i, j, 0], img[i, j, 1], img[i, j, 2]])
            rad = np.linalg.norm(col)
            # normalize the color
            col = col / rad if rad > 0 else np.array([0, 0, 0])
            # computr hue angle
            angle = np.arctan2(-col[1], -col[0]) / np.pi
            # interpolate color form the colorwheel
            idx = np.floor(angle * (ncols - 1) / 2)
            idx = np.mod(idx, ncols)
            # deal with the case when idx is NaN
            if np.isnan(idx):
                flow[i, j, 0] = 0
                flow[i, j, 1] = 0
                continue
            else: idx = int(idx)
            col0 = colorwheel[idx]
            col1 = colorwheel[int(np.mod(idx + 1, ncols))]
            f = angle * ncols / 2 - idx
            for k in range(2):
                flow[i, j, k] = (1 - f) * col0[k] + f * col1[k]
            flow[i, j, 0] = rad * (2 * flow[i, j, 0] - 1)
            flow[i, j, 1] = rad * (2 * flow[i, j, 1] - 1)
    
    return flow


UNKNOWN_FLOW_THRESH = 1e9
UNKNOWN_FLOW = 1e10


def flow_to_color(flow, max_flow=None):
    """
    Converts optical flow to a color image.

    Args:
        flow: A NumPy array of shape (height, width, 2) representing the optical flow.
                The first channel corresponds to horizontal (u) motion, 
                and the second channel corresponds to vertical (v) motion.
        max_flow: Optional maximum flow value for normalization. If None,
                normalization will be based on the maximum flow in the data.

    Returns:
        A NumPy array of shape (height, width, 3) representing the color-coded
        optical flow image.
    """

    height, width, n_bands = flow.shape

    if n_bands != 2:
        raise ValueError("flow must have two bands (u, v)")

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    max_u = np.max(u)  
    max_v = np.max(v)
    min_u = np.min(u)
    min_v = np.min(v)
    max_rad = -1

    # Fix unknown flow
    unknown_mask = (np.abs(u) > UNKNOWN_FLOW_THRESH) | (np.abs(v) > UNKNOWN_FLOW_THRESH)
    u[unknown_mask] = 0
    v[unknown_mask] = 0

    max_u = np.max([max_u, np.max(u)])
    min_u = np.min([min_u, np.min(u)])
    max_v = np.max([max_v, np.max(v)])
    min_v = np.min([min_v, np.min(v)])

    rad = np.linalg.norm(flow, axis=2)
    max_rad = np.max(rad)

    # print(f"Max flow: {max_rad:.4f}, Flow range: u = {min_u:.3f} .. {max_u:.3f}, v = {min_v:.3f} .. {max_v:.3f}")

    if max_flow is not None and max_flow > 0:
        max_rad = max_flow

    # Normalize
    u = u / (max_rad + np.spacing(1))  # Avoid division by zero with epsilon
    v = v / (max_rad + np.spacing(1))

    # Call computeColor function (assumed to be implemented separately)
    img = compute_color(u, v)

    # Set unknown flow pixels to black
    img[unknown_mask] = 0

    return img


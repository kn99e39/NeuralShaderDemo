"""u* for the REINIT ownership test: the P16 tea-pot (teapot4) unit with the most training pixels."""

import json
import sys

import numpy as np

z = np.load(sys.argv[1])            # partition_P16.npz
tr = json.load(open(sys.argv[2]))   # P16 train.json
counts = np.array(tr["owner_pixel_counts_train"])
cand = [u for u in range(int(z["units"])) if int(z["unit_part"][u]) == 3]  # PARTS order: plate, teapot2, teapot3, teapot4
print(max(cand, key=lambda u: counts[u]))

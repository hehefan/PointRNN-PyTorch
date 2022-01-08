import torch
import MinkowskiEngine as ME

data = [
    [0, 0, 2.1, 0, 0],
    [0, 1, 1.4, 3, 0],
    [0, 0, 4.0, 0, 0]
]

def to_sparse_coo(data):
    # An intuitive way to extract coordinates and features
    coords, feats = [], []
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            if val != 0:
                coords.append([i, j])
                feats.append([val])
    return torch.IntTensor(coords), torch.FloatTensor(feats)

coords, feats = to_sparse_coo(data)


st = ME.SparseTensor(coordinates=coords, features=feats)

print(st.dense())

import numpy as np
import torch
from numba import njit


@njit(cache=True)
def maximum_path_c(paths, values, frame_lengths, text_lengths):
  # Same dynamic program as the upstream Cython extension, without a C++ compiler.
  for b in range(values.shape[0]):
    frames, texts = frame_lengths[b], text_lengths[b]
    for y in range(frames):
      for x in range(max(0, texts + y - frames), min(texts, y + 1)):
        stay = values[b, y-1, x] if y > 0 and x < y else -1e9
        prev = values[b, y-1, x-1] if y > 0 and x > 0 else (0.0 if y == 0 and x == 0 else -1e9)
        values[b, y, x] += max(stay, prev)
    x = texts - 1
    for y in range(frames - 1, -1, -1):
      paths[b, y, x] = 1
      if x > 0 and (x == y or values[b, y-1, x] < values[b, y-1, x-1]):
        x -= 1


def maximum_path(neg_cent, mask):
  """ Cython optimized version.
  neg_cent: [b, t_t, t_s]
  mask: [b, t_t, t_s]
  """
  device = neg_cent.device
  dtype = neg_cent.dtype
  neg_cent = neg_cent.data.cpu().numpy().astype(np.float32)
  path = np.zeros(neg_cent.shape, dtype=np.int32)

  t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
  t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
  maximum_path_c(path, neg_cent, t_t_max, t_s_max)
  return torch.from_numpy(path).to(device=device, dtype=dtype)

import sys
import os
import time
import torch

# ensure repo dir is on sys.path
_proj = os.path.dirname(os.path.abspath(__file__))
if _proj not in sys.path:
    sys.path.insert(0, _proj)

from model_v2_encoder_cnn import ViTS16Encoder

print('Python:', sys.version.splitlines()[0])
print('Torch:', torch.__version__)

device = torch.device('cpu')
enc = ViTS16Encoder(encoder_dim=2048, pretrained=False).to(device)
enc.eval()

# test with non-224 input (256x256) to verify interpolation
x = torch.randn(2,3,256,256).to(device)
start = time.time()
with torch.no_grad():
    out = enc(x)
end = time.time()
print('Input shape:', x.shape)
print('Output shape:', out.shape)
print('Time (s):', end-start)


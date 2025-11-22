import torch
from model_v2_encoder_cnn import EncoderV2

print('PyTorch version:', torch.__version__)

def test_encoder(backbone):
    print('\nTesting', backbone)
    enc = EncoderV2(backbone=backbone, encoder_dim=512, pretrained=False)
    enc.eval()
    x = torch.randn(2,3,224,224)
    with torch.no_grad():
        out = enc(x)
    print('Output shape:', out.shape)

for b in ['efficientnet_b0','vit_small_patch16_224']:
    try:
        test_encoder(b)
    except Exception as e:
        print('Error testing', b, ':', repr(e))


from . import mlp
from . import gaussian


def get_model(model_type, in_dim, out_dim, device, **kwargs):
    if model_type == 'mlp':
        feat_dim = int(kwargs['feat_dim'][0])
        model = mlp.Model(in_dim, out_dim, feat_dim)
    if model_type == 'gaussian':
        feat_dim = int(kwargs['feat_dim'][0])
        model = gaussian.Model(in_dim, out_dim, feat_dim)
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    return model.to(device).eval()

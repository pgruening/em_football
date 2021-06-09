import copy

import torch
from DLBio import pt_training
from DLBio.helpers import dict_to_options
from DLBio.kwargs_translator import get_kwargs
from DLBio.pytorch_helpers import load_model_with_opt, get_device

from models.model_getter import get_model


def load_model(options, device, strict=True, new_model_path=None,
               map_location=None, from_par_gpu=False):
    if isinstance(options, dict):
        options = dict_to_options(options)

    def get_model_fcn(options, device):
        model_kwargs = get_kwargs(options.model_kw)
        return get_model(
            options.model_type,
            options.in_dim,
            options.out_dim,
            device,
            **model_kwargs
        )

    if new_model_path is not None:
        model_path = new_model_path
    else:
        model_path = options.model_path

    return load_model_with_opt(
        model_path,
        options,
        get_model_fcn,
        device,
        strict=strict,
        map_location=map_location,
        from_par_gpu=from_par_gpu
    )


def predict_needed_gpu_memory(options, *, input_shape, device, factor=2.1):
    # changes to the options object here should not be saved to the actual object
    options = copy.deepcopy(options)

    if isinstance(options, dict):
        # initialize a new model
        options['model_path'] = None
        options = dict_to_options(options)

    if device is not None:
        pt_training.set_device(device, verbose=False)

    model = load_model(options, get_device())

    torch.cuda.reset_max_memory_allocated()
    for _ in range(3):
        x = torch.rand(*input_shape).to(get_device())
        model(x)

    fwd_memory_used = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()

    # return as MegaByte
    return factor * fwd_memory_used / 1e6

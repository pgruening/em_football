import argparse
import json
from os.path import join

from DLBio import pt_training
from DLBio.helpers import check_mkdir
from DLBio.kwargs_translator import get_kwargs
from DLBio.pt_train_printer import Printer
from DLBio.pytorch_helpers import get_device, get_num_params, save_options
from DLBio.train_interfaces import get_interface

from helpers import load_model
from data.data_getter import get_data_loaders


PRINT_FREQUENCY = 50


def get_options():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--ds_kwargs', type=str, default=None)

    # training hyperparameters
    parser.add_argument('--opt', type=str, default=None)
    parser.add_argument('--nw', type=int, default=-1)
    parser.add_argument('--bs', type=int, default=-1)
    parser.add_argument('--lr', type=float, default=-1)
    parser.add_argument('--wd', type=float, default=-1)
    parser.add_argument('--input_dim', type=int, default=-1)
    parser.add_argument('--output_dim', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=-1)
    parser.add_argument('--lr_steps', type=int, default=-1)

    # train interface hyperparameters
    parser.add_argument('--train_interface', type=str, default=None)
    parser.add_argument('--ti_kwargs', type=str, default=None)

    # early stopping
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--es_get_min', action='store_false')
    parser.add_argument('--es_metric', type=str, default=None)

    # model saving
    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--sv_int', type=int, default=-1)

    # model definition
    parser.add_argument('--model_kw', type=str, default=None)
    parser.add_argument('--model_type', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)

    # other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=None)

    return parser.parse_args()


def run(options):
    if options.device is not None:
        pt_training.set_device(options.device)
    device = get_device()

    pt_training.set_random_seed(options.seed)

    # NOTE: full folder path needed
    folder = options.folder
    check_mkdir(folder, is_dir=True)

    # "load" model from scratch
    model = load_model(options, device)

    model_out = join(folder, 'model.pt')
    log_file = join(folder, 'log.json')

    save_options(join(folder, 'opt.json'), options)

    # write some model specs
    with open(join(folder, 'model_specs.json'), 'w') as file:
        json.dump({
            'num_trainable': float(get_num_params(model, True)),
            'num_params': float(get_num_params(model, False))
        }, file)

    optimizer = pt_training.get_optimizer(
        options.opt, model.parameters(),
        options.lr,
        weight_decay=options.wd
    )

    if options.lr_steps > 0:
        scheduler = pt_training.get_scheduler(
            options.lr_steps, options.epochs, optimizer)
    else:
        scheduler = None

    if options.early_stopping:
        assert options.sv_int == -1
        early_stopping = pt_training.EarlyStopping(
            options.es_metric, get_max=not options.es_get_min,
            epoch_thres=options.epochs
        )
    else:
        early_stopping = None

    printer = Printer(PRINT_FREQUENCY, log_file)

    train_interface = get_interface(
        options.train_interface,
        model, device, Printer(options.print_freq, log_file),
        num_epochs=options.epochs,
        **get_kwargs(options.ti_kwargs)
    )
    data_loaders = get_data_loaders(
        options.dataset, options.bs, options.nw, **get_kwargs(
            options.ds_kwargs)
    )

    training = pt_training.Training(
        optimizer, data_loaders['train'], train_interface,
        scheduler=scheduler, printer=printer,
        save_path=model_out, save_steps=options.sv_int,
        val_data_loader=data_loaders['val'],
        early_stopping=early_stopping,
        test_data_loader=data_loaders['test']
    )

    training(options.epochs)


if __name__ == '__main__':
    OPTIONS = get_options()
    run(OPTIONS)

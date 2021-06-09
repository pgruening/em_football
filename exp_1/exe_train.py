import copy
import itertools
from os.path import join

from DLBio import kwargs_translator, pt_run_parallel
from helpers import predict_needed_gpu_memory

DEFAULT_KWARGS = {
    'comment': 'first test',
    'lr': 0.01,
    'wd': 0.01,
    'mom': 0.,
    'bs': 128,
    'opt': 'Adam',

    'train_interface': 'MatchAndGoals',

    # model / ds specific params
    'in_dim': 55,
    'out_dim': -1,

    # scheduling
    'epochs': 20,
    'lr_steps': 0,

    # dataset
    'dataset': 'qualification',

    # model saving
    'sv_int': -1,
    'early_stopping': None,
    'es_metric': 'val_loss',


}
# 12 10 2020

DATASET_KWARGS = {
    'year': 2020,
    'month': 10,
    'day': 12
}


EXE_FILE = 'run_training.py'
BASE_FOLDER = 'experiments/exp_1'

MODELS = ['mlp']

AVAILABLE_GPUS = [0]
SEEDS = [42]


LRS = [.1, .01, .001, .0001, .00001]
OPTS = ['Adam', 'SGD', 'lamb', 'AdaDelta']
WDS = [0., 0.1, 0.001, 0.0001, 0.00001]
ANGLES = [0., 0.01, 0.05, 0.1, .2, .499]


class TrainingProcess(pt_run_parallel.ITrainingProcess):
    def __init__(self, **kwargs):
        self.start_time = -1
        self.device = -1
        self.mem_used = None

        self.__name__ = 'Exp. 1 training process'
        self.module_name = EXE_FILE
        self.kwargs = kwargs


def _run(param_generator):
    make_object = pt_run_parallel.MakeObject(TrainingProcess)
    pt_run_parallel.run_bin_packing(
        param_generator(), make_object,
        available_gpus=AVAILABLE_GPUS,
        log_file=join(
            BASE_FOLDER, 'parallel_train_log.txt'),
        shuffle_params=True
    )


def run():
    default_kwargs = copy.deepcopy(DEFAULT_KWARGS)
    base_folder = join(BASE_FOLDER, 'exp_data')

    def param_generator():
        for p in _param_generator(default_kwargs, base_folder):
            yield p

    _run(param_generator)


def _param_generator(default_kwargs, base_folder, seeds=SEEDS):
    train_params = itertools.product(LRS, OPTS, WDS, ANGLES)
    ctr = 0
    for model in MODELS:
        for seed in seeds:
            for lr, opt, wd, mxa in train_params:
                output = copy.deepcopy(default_kwargs)
                output['model_type'] = model
                output['folder'] = join(base_folder, str(ctr).zfill(4))
                output['seed'] = seed

                output['lr'] = lr
                output['opt'] = opt
                output['wd'] = wd

                model_kw = {'max_angle': [mxa]}
                output['model_kw'] = kwargs_translator.to_kwargs_str(model_kw)

                # add expected memory usage for bin-packing
                output['mem_used'] = predict_needed_gpu_memory(
                    output, input_shape=(
                        output['bs'], output['in_dim']),
                    device=AVAILABLE_GPUS[0]
                )

                ctr += 1

                yield output


if __name__ == '__main__':
    run()

from DLBio.helpers import MyDataFrame, search_rgx, load_json
from os.path import join
from DLBio.kwargs_translator import get_kwargs

PATH = 'experiments/exp_2/exp_data'


def run():
    df = MyDataFrame()
    for folder in search_rgx(r'\d\d\d\d', PATH):
        df = update(join(PATH, folder), df)

    df = df.get_df()
    #df = df.sort_values(['val_acc', 'val_mse'], ascending=False)
    df = df.sort_values(['val_abs'], ascending=False)
    print(df)


def update(folder, df):
    opt = load_json(join(folder, 'opt.json'))
    log = load_json(join(folder, 'log.json'))

    model_kw = get_kwargs(opt['model_kw'])

    if log is None:
        return df

    df.update({
        'id': folder.split('/')[-1],
        'lr': opt['lr'],
        'acc': log['acc'][-1],
        'abs': log['abs'][-1],
        'loss': -1. * log['loss'][-1],
        'val_acc': log['val_acc'][-1],
        'val_abs': -1. * log['val_abs'][-1],
        'feat_dim': int(model_kw['feat_dim'][0])
    })

    return df


if __name__ == '__main__':
    run()

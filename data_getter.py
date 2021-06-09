from . import ds_qualification


def get_data_loaders(dataset_type, batch_size, num_workers, **kwargs):
    if dataset_type == 'qualification':
        val_after = {
            'year': int(kwargs['year'][0]),
            'month': int(kwargs['month'][0]),
            'day': int(kwargs['day'][0]),
        }
        return {
            'train': ds_qualification.get_data_loader(
                batch_size, num_workers, is_train=True, validation_after_date=val_after
            ),
            'val': ds_qualification.get_data_loader(
                batch_size, num_workers, is_train=False, validation_after_date=val_after
            ),
            'test': None
        }

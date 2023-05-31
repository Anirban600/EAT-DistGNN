BEST_HPARAMS = {
    'flickr': {
        'g': {
            'fan_out': '25,25',
            'num_hidden': 256,
            'lr': 0.001,
            'gamma': 0.15
        },
        0: {
            'fan_out': '25,25',
            'llambda': 0.00001,
            'lr': 0.003,
            'gamma': 0.1
        },
        1: {
            'fan_out': '25,10',
            'llambda': 0.0001,
            'lr': 0.001,
            'gamma': 0.1
        },
        2: {
            'fan_out': '25,10',
            'llambda': 0.000001,
            'lr': 0.003,
            'gamma': 0.1
        },
        3: {
            'fan_out': '25,10',
            'llambda': 0.000001,
            'lr': 0.003,
            'gamma': 0.1
        }
    }
}

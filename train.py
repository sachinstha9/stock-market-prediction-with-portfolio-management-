import argparse

from stockdash.config import DEFAULT_TICKERS, MODEL_PATH
from stockdash.data import download_history, make_supervised
from stockdash.modeling import train_and_save, model_candidates


def main():
    p = argparse.ArgumentParser(
        description='Train and validate a multi-stock classifier on real Yahoo Finance history.'
    )

    p.add_argument('--period', default='10y')
    p.add_argument('--horizon', type=int, choices=[1, 5, 10, 20], default=5)
    p.add_argument('--mode', choices=['UP/DOWN', 'UP/DOWN/SIDEWAYS'], default='UP/DOWN')
    p.add_argument('--tickers', nargs='*', default=None,
                   help='Optional symbols; default is multi-sector universe in config.py')
    p.add_argument('--min-balanced-accuracy', type=float, default=0.48)
    p.add_argument('--model', choices=['AUTO'] + list(model_candidates().keys()),
                   default='AUTO',
                   help='AUTO compares all models; otherwise train only the selected model')

    args = p.parse_args()

    tickers = args.tickers or [
        t for sector in DEFAULT_TICKERS.values() for t in sector
    ]

    print(f'Downloading real data for {len(tickers)} symbols, period={args.period}...')

    prices = download_history(tickers, period=args.period)

    print('Usable symbols:', ', '.join(prices))

    dataset = make_supervised(prices, args.horizon, args.mode)

    print(
        f'Supervised rows={len(dataset):,}; '
        f'unique dates={dataset.index.nunique():,}; '
        f'classes={dataset.target.value_counts().to_dict()}'
    )

    selected = None if args.model == 'AUTO' else [args.model]

    result = train_and_save(
        dataset,
        args.horizon,
        args.mode,
        args.min_balanced_accuracy,
        MODEL_PATH,
        selected_models=selected
    )

    print('\nWalk-forward model comparison:')
    print(result['scores'].drop(columns=['fold_details']).to_string(index=False))

    print('\n', result['reason'])

    if result['saved']:
        print('Saved artifact:', MODEL_PATH)
    else:
        print('Existing model artifact was not replaced.')


if __name__ == '__main__':
    main()
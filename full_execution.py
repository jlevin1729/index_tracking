import torch
import pickle
from stock_subset.reinforce import reinforce
from weights.genetic_algo import genetic_algorithm, tracking_error
import pylightxl as xl
from typing import Tuple, Optional, List
import heapq
from pathlib import Path
from matplotlib import pyplot as plt
from plots import plot_tracking_error, plot_mean_tracking_error

db = xl.readxl(fn='/Users/joshlevin/PycharmProjects/Santander_P2/data/dataset.xlsx')

UK = 'MSCI_UK_val'
xUK = 'MSCI_Europe_xUK_val'
USA = 'MSCI_USA_val'

UK_stocks = (torch.tensor(db.ws(ws=UK).range(address='B2:CN360')), 'UK')
#xUK_stocks = (torch.tensor(db.ws(ws=xUK).range(address='B2:MJ360')), 'xUK')
USA_stocks = (torch.tensor(db.ws(ws=USA).range(address='B2:WR360')), 'USA')


def index_tracking_port(
    run_name: str,
    market: Tuple[torch.Tensor, str],
    day: int,
    num_epochs: int,
    rate: float,
    min_rate: float,
    gamma_stocks: float,
    gamma_weights: float,
    layer_dims: List[int],
    portfolio_frac: float,
    init_pop_size: int,
    cull_frac: float,
    quantum: Optional[bool] = False,
    error_goal: Optional[float] = None,
    num_generations: Optional[int] = None
):
    base_dir = Path('/Users/joshlevin/PycharmProjects/Santander_P2/runs')
    main_dir = base_dir / Path(run_name)
    stock_dir = main_dir / Path('stock_selection')
    weights_dir = main_dir / Path('weight_selection')
    main_dir.mkdir(parents=True)
    stock_dir.mkdir()
    weights_dir.mkdir()

    #with '/Users/joshlevin/PycharmProjects/Santander_P2/runs'
    print('Beginning autoencoder training')
    autoencoder = reinforce(
        run_name=run_name,
        day=day,
        market=market,
        num_epochs=num_epochs,
        rate=rate,
        min_rate=min_rate,
        gamma=gamma_stocks,
        layer_dims=layer_dims,
        quantum=quantum
    )

    print('Finding best stocks')

    if market[1] == 'UK':
        orig = market[0][:day, :-2]
    else:
        orig = market[0][:day, :-1]
    recon = autoencoder.forward(orig)
    diff = (recon - orig) / orig
    scores = torch.mean(diff ** 2, dim=0)

    num_stocks = int(portfolio_frac * (market[0].size()[1] - 1))
    if market[1] == 'UK':
        winners = [89] + [list(scores).index(i) for i in heapq.nsmallest(num_stocks - 1, scores)]
    else:
        winners = [list(scores).index(i) for i in heapq.nsmallest(num_stocks, scores)]

    with open('/Users/joshlevin/PycharmProjects/Santander_P2/runs/' + run_name + '/stock_selection/selected_stocks.py', 'wb') as f:
        pickle.dump(winners, f)

    print()
    print("Beginning weight optimization")
    portfolio = genetic_algorithm(
        run_name=run_name,
        market=market,
        day=day,
        portfolio_frac=portfolio_frac,
        init_pop_size=init_pop_size,
        cull_frac=cull_frac,
        error_goal=error_goal,
        num_generations=num_generations,
        stocks=winners,
        gamma=gamma_weights
    )

    mean_errors = []
    for end in range(day + 1, 360):
        error = tracking_error(portfolios=portfolio, market=market[0], day=end, start_day=day)
        mean_errors.append(error)
    print('mean_errors =', mean_errors)
    print('length of mean_errors =', len(mean_errors))
    plot_mean_tracking_error(save_dir=main_dir, errors=mean_errors)

    errors = []
    for test_day in range(day, 359):
        error = tracking_error(portfolios=portfolio, market=market[0], day=test_day + 1, start_day=test_day, daily=True)
        errors.append(float(error))
    print('errors =', errors)
    print('length of errors =', len(errors))
    plot_tracking_error(save_dir=main_dir, errors=errors)

    print()
    print('Done')
    return portfolio


if __name__ == '__main__':
    """
    for day in range(1, 113):
        index_tracking_port(
            run_name='output_runs_2/UK_day' + str(day) + '_noq',
            market=UK_stocks,
            day=day,
            num_epochs=200000,
            rate=0.2,
            min_rate=0.00001,
            gamma_stocks=0.9999,
            gamma_weights=1.,
            layer_dims=[90, 40, 16, 40, 90],
            portfolio_frac=0.2,
            init_pop_size=10000,
            cull_frac=0.01,
            num_generations=100,
            quantum=False
        )
    """

    for day in range(2, 113):
        index_tracking_port(
            run_name='output_runs_3/USA_day' + str(day) + '_noq',
            market=USA_stocks,
            day=day,
            num_epochs=300000,
            rate=0.2,
            min_rate=0.000001,
            gamma_stocks=0.9999458,
            gamma_weights=0.985,
            layer_dims=[614, 360, 200, 120, 200, 360, 614],
            portfolio_frac=0.2,
            init_pop_size=10000,
            cull_frac=0.01,
            num_generations=200,
            quantum=False
        )
    """
    index_tracking_port(
        run_name='UK_test5',
        market=UK_stocks,
        day=100,
        num_epochs=100000,
        rate=0.2,
        min_rate=0.00001,
        gamma_stocks=0.9999,
        gamma_weights=0.993,
        layer_dims=[89, 65, 40, 20, 40, 65, 89],
        portfolio_frac=0.2,
        init_pop_size=10000,
        cull_frac=0.01,
        num_generations=300,
        quantum=False
    )
"""

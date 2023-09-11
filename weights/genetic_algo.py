import pylightxl as xl
import torch
import numpy as np
import random as rdm
import heapq
from typing import List, Optional, Tuple
from copy import copy
import time
import pickle
from pathlib import Path
from plots import plot_min_errors

db = xl.readxl(fn='data/dataset.xlsx')

UK = 'MSCI_UK_val'
xUK = 'MSCI_Europe_xUK_val'
USA = 'MSCI_USA_val'

UK_stocks = (torch.tensor(db.ws(ws=UK).range(address='B2:CN360')), 'UK')
#xUK_np = np.array(db.ws(ws=xUK).range(address='B2:MJ360'))
#xUK_np[:106, ]
#xUK_stocks = (torch.tensor(db.ws(ws=xUK).range(address='B2:MJ360')), 'xUK')
USA_stocks = (torch.tensor(db.ws(ws=USA).range(address='B2:WR360')), 'USA')


def random_sparse_portfolio(market_size: int, portfolio_size: int, bounds: List[Tuple[float, float]], stocks: Optional[List[int]] = None) -> torch.tensor:
    market = range(market_size)
    if stocks is None:
        included = rdm.sample(market, portfolio_size)
    else:
        included = stocks
    portfolio = torch.tensor([rdm.uniform(bounds[i][0], bounds[i][1]) if i in included else 0 for i in market])
    portfolio = portfolio / sum(portfolio)
    if not np.isclose(sum(portfolio), 1):
        raise RuntimeError(f'Portfolio weights sum to {sum(portfolio)}')
    return portfolio


def weight_bounds(market: torch.tensor, day: int) -> List[Tuple[float, float]]:
    bounds = []
    market_data = market[:day, :-1]
    index_data = market[:day, -1:]
    upper_bounds_ = index_data / market_data
    upper_bounds = upper_bounds_.transpose(0, 1)
    for stock in range(upper_bounds.shape[0]):
        min_upper = min(upper_bounds[stock])
        bounds.append((0.002, min(0.06, min_upper)))
    return bounds


def initialize_population(market_size: int,
                          portfolio_size: int,
                          pop_size: int,
                          weight_bds: List[Tuple[float, float]],
                          stocks: Optional[List[int]] = None
                          ) -> torch.tensor:
    population_ = [random_sparse_portfolio(
        market_size=market_size,
        portfolio_size=portfolio_size,
        bounds=weight_bds,
        stocks=stocks) for i in range(pop_size)]
    population = torch.stack(population_, 1)
    return population


def tracking_error(portfolios: torch.tensor, market: torch.tensor, day: int, start_day: Optional[int] = None, daily: bool = False) -> torch.tensor:
    if start_day:
        market_data = market[start_day:day, :-1]
        index_returns = market[start_day:day, -1:]
    else:
        market_data = market[:day, :-1]
        index_returns = market[:day, -1:]
    market_returns = torch.matmul(market_data, portfolios)
    diff = (market_returns - index_returns) / index_returns
    #print('diff =', diff)
    if daily:
        return diff.squeeze(0)
    else:
        return torch.mean(diff ** 2, dim=0).sqrt()


def stocks_included(portfolio: torch.tensor) -> set:
    return portfolio.nonzero()[0]


def random_mutation(portfolios: torch.tensor, num_mutants: int, bounds: List[Tuple[float, float]], step_size: float, ) -> torch.tensor:
    expanded = [portfolios[:, i: i + 1].expand((-1, num_mutants)) for i in range(portfolios.size()[1])]
    stacked = torch.cat(expanded, 1)

    #included = [portfolios[:, i].nonzero() for i in range(portfolios.size()[1])]
    #included_sets = [{tensor.item() for tensor in list(A)} for A in included]
    #excluded_sets = [set(range(portfolios.size()[0])) - inc for inc in included_sets]
    #random_zeros = [rdm.choices(list(s), k=int(num_mutants / 2)) for s in included_sets]
    #random_swaps = [rdm.choices(list(s), k=int(num_mutants / 2)) for s in excluded_sets]

    #zero_mask = torch.ones(stacked.size())
    #for i in range(len(random_zeros)):
    #    for j in range(len(random_zeros[i])):
    #        row = list(random_zeros[i])[j]
    #        column = i * num_mutants + j
    #        zero_mask[row][column] = 0

    #with_removed = stacked * zero_mask

    #add = torch.zeros(stacked.size())
    #for i in range(len(random_swaps)):
    #    for j in range(len(random_swaps[i])):
    #        row = list(random_swaps[i])[j]
    #        column = i * num_mutants + j
    #        add[row][column] = rdm.uniform(bounds[row][0], bounds[row][1])

    #swapped = with_removed + add

    weight_adjustments = torch.normal(torch.ones(stacked.size()), step_size * torch.ones(stacked.size()))
    #print('weight adjustments size =', weight_adjustments.size())
    mutants = (stacked * weight_adjustments).squeeze()
    return mutants / mutants.sum(0)


"""
    options = [(1, 0), (0, 1), (1, 1)]
    mutation_type = rdm.sample(options, 1)[0]
    exchange_one_stock = mutation_type[0]
    adjust_one_weight = mutation_type[1]

    included = stocks_included(portfolio=portfolio_vec)
    excluded = set(range(portfolio.shape[0])) - included

    if exchange_one_stock:
        stock_to_remove = rdm.sample(list(included), 1)[0]
        stock_to_add = rdm.sample(list(excluded), 1)[0]

        portfolio_vec[stock_to_add] = rdm.uniform(bounds[stock_to_add][0], bounds[stock_to_add][1])
        portfolio_vec[stock_to_remove] = 0

        portfolio_vec = portfolio_vec / sum(portfolio_vec)
        del stock_to_add, stock_to_remove

    if adjust_one_weight:
        weight_to_adjust = rdm.sample(list(stocks_included(portfolio=portfolio_vec)), 1)[0]
        current_value = portfolio_vec[weight_to_adjust]
        direction = rdm.randint(0, 1)
        if direction:
            new_value = current_value + step_size * (bounds[weight_to_adjust][1] - current_value)
        else:
            new_value = current_value - step_size * (current_value - bounds[weight_to_adjust][0])
        portfolio_vec[weight_to_adjust] = new_value
        portfolio_vec = portfolio_vec / sum(portfolio_vec)
        if not np.isclose(sum(portfolio_vec), 1):
            raise RuntimeError(f'Mutated portfolio weights sum to {sum(portfolio)}')
    return portfolio_vec
"""


def genetic_algorithm(
        run_name: str,
        market: Tuple[torch.tensor, str],
        day: int,
        portfolio_frac: float,
        init_pop_size: int,
        cull_frac: float,
        gamma: float,
        error_goal: Optional[float] = None,
        num_generations: Optional[int] = None,
        stocks: Optional[List[int]] = None
) -> (torch.tensor, List[float]):
    run_params = {'market': market[1],
                  'day': day,
                  'port frac': portfolio_frac,
                  'pop size': init_pop_size,
                  'cull frac': cull_frac,
                  'error goal': error_goal,
                  'num gens': num_generations}
    with open('/Users/joshlevin/PycharmProjects/Santander_P2/runs/' + run_name + '/weight_selection/params.py', 'wb') as f:
        pickle.dump(run_params, f)
    generation = 0
    min_error = 1e17
    smallest_error = []
    best_portfolio = 0

    if not error_goal and num_generations:
        error_goal = 0
    elif not num_generations and error_goal:
        num_generations = 1e17
    elif not error_goal and not num_generations:
        raise RuntimeError('Genetic algorithm needs an error goal, a number of generations, '
                           'or both, but received neither')
    else:
        pass

    market_size = market[0].shape[1] - 1
    portfolio_size = round(portfolio_frac * market_size)

    bounds = weight_bounds(market=market[0], day=day)

    print('Initializing population')
    population = initialize_population(market_size=market_size,
                                       portfolio_size=portfolio_size,
                                       pop_size=init_pop_size,
                                       weight_bds=bounds,
                                       stocks=stocks)

    print('Beginning evolution')
    while generation < num_generations and min_error > error_goal:
        start = time.time()
        # print('population =', population)
        print()
        print('generation', generation)
        pop_tracking_errors = tracking_error(portfolios=population.float(), market=market[0], day=day)
        # print('errors =', pop_tracking_errors)
        smallest_error_in_generation = torch.min(pop_tracking_errors)
        if smallest_error_in_generation < min_error:
            best_index = torch.argmin(pop_tracking_errors)
            best_portfolio = population[:, best_index].unsqueeze(1)
            with open('/Users/joshlevin/PycharmProjects/Santander_P2/runs/' + run_name + '/weight_selection/portfolio.py', 'wb+') as f:
                pickle.dump(best_portfolio, f)
        min_error = smallest_error_in_generation
        smallest_error.append(min_error)
        with open('/Users/joshlevin/PycharmProjects/Santander_P2/runs/' + run_name + '/weight_selection/min_errors.py', 'wb+') as f:
            pickle.dump(smallest_error, f)
        print('min error = ', min_error)
        print('Breeding new generation')
        if min_error > error_goal:
            keep_number = round(population.size()[1] * cull_frac)
            heap_start = time.time()
            keep_indices = [list(pop_tracking_errors).index(i) for i in heapq.nsmallest(keep_number, pop_tracking_errors)]
            heap_end = time.time()
            print('heap time =', heap_end - heap_start)
            culled = population[:, keep_indices]
            mut_start = time.time()
            mutants = random_mutation(culled, int((1 / cull_frac) - 1), bounds=bounds, step_size=0.1 * (gamma ** generation))
            mut_end = time.time()
            print('mutation time =', mut_end - mut_start)

            population = torch.cat([culled, mutants], 1)
            # if best_portfolio not in population:
            #    raise RuntimeError('Breeder did not make it to next generation')
            # else:
            #    print('breeder from previous generation =', population.index(best_portfolio))
            generation += 1

        save_dir = Path('/Users/joshlevin/PycharmProjects/Santander_P2/runs/' + run_name + '/weight_selection')
        plot_min_errors(save_dir=save_dir, min_errors=smallest_error)

        end = time.time()
        print('time = ', end - start)
    return best_portfolio


if __name__ == '__main__':

    result = genetic_algorithm(run_name='test_run',
                               market=USA_stocks,
                               day=100,
                               portfolio_frac=0.2,
                               init_pop_size=1000000,
                               cull_frac=0.001,
                               num_generations=1000)

    #port = torch.tensor([[1, 1], [1, 0], [0, 0], [0, 1], [1, 1], [0, 0], [0, 0]])
    #B = random_mutation(portfolios=port, num_mutants=4, bounds=[(2,4), (3,6), (17,18), (5,6), (0,1), (10, 12), (100,102)])
    #print(B)

    #file = open('USA_day100_frac20.py', 'rb')
    #params = pickle.load(file)
    #result = pickle.load(file)
    #port = result[0]
    #print(port)
    #print(tracking_error(port, USA_stocks[0], 359, 100))

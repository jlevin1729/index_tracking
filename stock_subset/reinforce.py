import torch
import pickle
from typing import List, Tuple, Optional
import pylightxl as xl
from stock_subset.auto_encoder import AutoEncoder, QuantumAutoEncoder
from pathlib import Path
from plots import plot_epoch_losses

db = xl.readxl(fn='/Users/joshlevin/PycharmProjects/Santander_P2/data/dataset.xlsx')

UK = 'MSCI_UK_val'
xUK = 'MSCI_Europe_xUK_val'
USA = 'MSCI_USA_val'

UK_stocks = (torch.tensor(db.ws(ws=UK).range(address='B2:CN360')), 'UK')
#xUK_stocks = (torch.tensor(db.ws(ws=xUK).range(address='B2:MJ360')), 'xUK')
USA_stocks = (torch.tensor(db.ws(ws=USA).range(address='B2:WR360')), 'USA')


def cost(reconstructed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    diff = reconstructed - original
    return sum(torch.norm(diff, p=2, dim=1))


def reinforce(
    run_name: str,
    day: int,
    market: Tuple[torch.Tensor, str],
    num_epochs: int,
    rate: float,
    min_rate: float,
    gamma: float,
    layer_dims: List[int],
    quantum: Optional[bool] = False
):
    run_params = {
        'run_name': run_name,
        'day': day,
        'market': market[1],
        'num_epochs': num_epochs,
        'rate': rate,
        'min_rate': min_rate,
        'gamma': gamma,
        'layer_dims': layer_dims
    }
    with open('/Users/joshlevin/PycharmProjects/Santander_P2/runs/' + run_name + '/stock_selection/params.py', 'wb') as f:
        pickle.dump(run_params, f)

    if not quantum:
        network = AutoEncoder(dims=layer_dims)
    else:
        network = QuantumAutoEncoder(dims=layer_dims)
    optimizer = torch.optim.Adam(params=network.parameters(), lr=rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

    if market[1] == 'UK':
        training_data = market[0][:day, :-2]
    else:
        training_data = market[0][:day, :-1]
    epoch_losses = []

    for epoch in range(num_epochs):
        if epoch % 10000 == 0:
            print()
            print('starting epoch', epoch)
        reconstructed = network.forward(training_data)
        loss = cost(reconstructed=reconstructed, original=training_data)
        epoch_losses.append(float(loss))
        if epoch % 10000 == 0:
            print('loss =', float(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print(network.state_dict())

        decayed_rate = rate * gamma ** (1 + epoch)
        if decayed_rate > min_rate:
            scheduler.step()
            if epoch % 10000 == 0:
                print('new rate =', decayed_rate)
        else:
            if epoch % 10000 == 0:
                print('rate locked at ', min_rate)
    reconstructed = network.forward(training_data)
    loss = cost(reconstructed=reconstructed, original=training_data)
    epoch_losses.append(float(loss))
    print()
    print('finished training')
    print('final loss =', float(loss))

    save_dir = Path('/Users/joshlevin/PycharmProjects/Santander_P2/runs/' + run_name + '/stock_selection')

    plot_epoch_losses(save_dir=save_dir, losses=epoch_losses[30000:])

    #print('final network =', network.state_dict())
    with open('/Users/joshlevin/PycharmProjects/Santander_P2/runs/' + run_name + '/stock_selection/model.pt', 'wb') as f:
        pickle.dump(network, f)
    with open('/Users/joshlevin/PycharmProjects/Santander_P2/runs/' + run_name + '/stock_selection/epoch_loss.py', 'wb') as f:
        pickle.dump(epoch_losses, f)
    return network


if __name__ == '__main__':

    AE = reinforce(
        run_name='USA_50d_100ke',
        day=50,
        market=USA_stocks,
        num_epochs=100000,
        rate=0.2,
        min_rate=0.001,
        gamma=0.999,
        layer_dims=[614, 200, 60, 200, 614]
    )

    orig = USA_stocks[0][:50, :-1]
    recon = AE.forward(orig)
    diff = (recon - orig) / orig
    sq = diff ** 2
    scores = torch.mean(sq, dim=0)
    print('stock scores =', scores)

import torch
import torch.nn as nn
import numpy as np
from sklearn.utils import shuffle
import tqdm
import random
import time

def train(
    net: torch.nn.Module,
    trainData,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
    client_id: int
) -> None:
    # Define loss and optimizer

    weight_decay = 1e-6
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.00001, weight_decay=weight_decay
    )

    # Train the network
    net.to(device)
    net.train()
    losses = []
    mean_loss = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        # trainData = shuffle(trainData)
        for i, data in enumerate(trainData):
            (
                batch_time,
                batch_lat,
                batch_lon,
                batch_delta_lat,
                batch_delta_lon,
                _,
            ) = data

            tmp_time = batch_time[1:]
            tmp_time = np.insert(tmp_time, 0, 0)
            batch_time = torch.FloatTensor(batch_time).reshape(-1, 1)
            batch_time_next = torch.FloatTensor(tmp_time).reshape(-1, 1)
            batch_lat = torch.FloatTensor(batch_lat).reshape(-1, 1)
            batch_lon = torch.FloatTensor(batch_lon).reshape(-1, 1)
            batch_delta_lat = torch.FloatTensor(batch_delta_lat).reshape(-1, 1)
            batch_delta_lon = torch.FloatTensor(batch_delta_lon).reshape(-1, 1)
            x = torch.cat(
                (
                    batch_time,
                    batch_lat,
                    batch_lon,
                    batch_delta_lat,
                    batch_delta_lon,
                    batch_time_next,
                ),
                1,
            ).split(5)
            rnn_input = torch.stack(x[:-1], 0)
            y = torch.cat((batch_delta_lat, batch_delta_lon), 1).split(5)
            rnn_y = torch.cat([z[0] for z in y], 0).reshape(-1, 2)[1:]

            rnn_input = rnn_input.to(device)
            rnn_y = rnn_y.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(rnn_input)
            loss = loss_fn(rnn_y, outputs)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        train_loss = np.mean(losses)
        mean_loss += train_loss/epochs
        losses.clear()
    # time.sleep(random.random())
    f = open("client_log.txt", "a")
    f.write(f'client_{client_id} TRAIN LOSS: {mean_loss}\n')
    f.close()


def test(
    net: torch.nn.Module,
    testData,
    device: torch.device,
    client_id: int,
    location_dict,
    step: bool
) -> None:

    loss_fn = nn.MSELoss(reduction="mean")

    # Train the network
    net.to(device)
    net.eval()
    losses = []

    predictions = []
    ref = []
    initial_location = []

    with torch.no_grad():
        for i, data in enumerate(testData):
            (
                batch_time,
                batch_lat,
                batch_lon,
                batch_delta_lat,
                batch_delta_lon,
                _,
            ) = data

            tmp_time = batch_time[1:]
            tmp_time = np.insert(tmp_time, 0, 0)
            batch_time = torch.FloatTensor(batch_time).reshape(-1, 1)
            batch_time_next = torch.FloatTensor(tmp_time).reshape(-1, 1)
            batch_lat = torch.FloatTensor(batch_lat).reshape(-1, 1)
            batch_lon = torch.FloatTensor(batch_lon).reshape(-1, 1)
            batch_delta_lat = torch.FloatTensor(batch_delta_lat).reshape(-1, 1)
            batch_delta_lon = torch.FloatTensor(batch_delta_lon).reshape(-1, 1)
            x = torch.cat(
                (
                    batch_time,
                    batch_lat,
                    batch_lon,
                    batch_delta_lat,
                    batch_delta_lon,
                    batch_time_next,
                ),
                1,
            ).split(5)
            rnn_input = torch.stack(x[:-1], 0)
            y = torch.cat((batch_delta_lat, batch_delta_lon), 1).split(5)
            rnn_y = torch.cat([z[0] for z in y], 0).reshape(-1, 2)[1:]

            rnn_input = rnn_input.to(device)
            rnn_y = rnn_y.to(device)

            # forward + backward + optimize
            outputs = net(rnn_input)
            loss = loss_fn(rnn_y, outputs)

            ref.append(rnn_y.cpu())
            predictions.append(outputs.cpu())
            initial_location.append(rnn_input[:, -1, 1:3].cpu())

            losses.append(loss.item())

        train_loss = np.mean(losses)
        losses.clear()
        
        f = open("client_log.txt", "a")
        f.write(f'client_{client_id} VALIDATION LOSS: {train_loss}\n')
        acc1 = acc5 = acc10 = 0
        if step:
            ref = torch.cat(ref, 0).reshape(-1, 2)
            predictions = torch.cat(predictions, 0).reshape(-1, 2)
            initial_location = torch.cat(initial_location, 0).reshape(-1, 2)

            min_lat = -0.04466687620000087
            min_lon = -0.04463771669999517

            max_lat = 0.04456799329999939
            max_lon = 0.04439055920000001

            ref[:, 0] = ref[:, 0] * (max_lat - min_lat) + min_lat
            ref[:, 1] = ref[:, 1] * (max_lon - min_lon) + min_lon

            predictions[:, 0] = predictions[:, 0] * (max_lat - min_lat) + min_lat
            predictions[:, 1] = predictions[:, 1] * (max_lon - min_lon) + min_lon

            min_lat = -42.90519725
            min_lon = -159.3360625167

            max_lat = 67.8650214
            max_lon = 174.8112130165

            initial_location[:, 0] = initial_location[:, 0] * (max_lat - min_lat) + min_lat
            initial_location[:, 1] = initial_location[:, 1] * (max_lon - min_lon) + min_lon

            
            ref[:, 0] = ref[:, 0] + initial_location[:, 0]
            ref[:, 1] = ref[:, 1] + initial_location[:, 1]

            predictions[:, 0] = predictions[:, 0] + initial_location[:, 0]
            predictions[:, 1] = predictions[:, 1] + initial_location[:, 1]

            keys = list(map(list, location_dict.keys()))
            keys = np.asarray(keys)
            b = [np.argmin(np.sum((keys - node.tolist())**2, axis=1)) for _, node in enumerate(tqdm.tqdm(ref, desc="ref"))]

            a = [True if b[i] in np.argpartition(np.sum((keys - node.tolist())**2, axis=1), 1)[:1] else False for i, node in enumerate(tqdm.tqdm(predictions, desc="predictions_1"))]
            acc1 = 0 
            for i, _ in enumerate(tqdm.tqdm(range(len(a)), desc = "acc")):
                if a[i]:
                    acc1 += 1
            f.write(f"{client_id}_acc_1: {acc1/len(ref)}\n")

            a = [True if b[i] in np.argpartition(np.sum((keys - node.tolist())**2, axis=1), 5)[:5] else False for i, node in enumerate(tqdm.tqdm(predictions, desc="predictions_5"))]
            acc5 = 0 
            for i, _ in enumerate(tqdm.tqdm(range(len(a)), desc = "acc")):
                if a[i]:
                    acc5 += 1
            f.write(f"{client_id}_acc_5: {acc5/len(ref)}\n")

            a = [True if b[i] in np.argpartition(np.sum((keys - node.tolist())**2, axis=1), 10)[:10] else False for i, node in enumerate(tqdm.tqdm(predictions, desc="predictions_10"))]
            acc10 = 0 
            for i, _ in enumerate(tqdm.tqdm(range(len(a)), desc = "acc")):
                if a[i]:
                    acc10 += 1
            f.write(f"{client_id}_acc_10: {acc10/len(ref)}\n")

            acc1 = acc1/len(ref)
            acc5 = acc5/len(ref)
            acc10 = acc10/len(ref)

        f.close()
        return train_loss, acc1, acc5, acc10

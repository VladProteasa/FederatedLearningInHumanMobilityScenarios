import sys

from dotenv import dotenv_values
from sklearn import cluster
from sklearn.utils import shuffle
from GRUmodel import GRUModel
from LSTMmodel import LSTMModel
from loader import Loader
from model import RNNModel
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import itertools
from heatmap import heatmap, corrplot
from sklearn.metrics import r2_score


ftype = torch.cuda.FloatTensor
ltype = torch.cuda.LongTensor

device = torch.device("cuda:0")


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    node = node.tolist()
    # print(node)
    # print(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

def main():

    config = dotenv_values(".env")
    # read data from file
    loader = Loader(config["dataset_file"])
    (
        users,
        times,
        lats,
        lons,
        delta_lats,
        delta_lons,
        locs,
        validate_times,
        validate_lats,
        validate_lons,
        validate_delta_lats,
        validate_delta_lons,
        validate_locs,
        test_times,
        test_lats,
        test_lons,
        test_delta_lats,
        test_delta_lons,
        test_locs,
        location_dict
    ) = loader.read_data()

    num_epochs = 400
    input_dim = 6
    hidden_dim = 32
    layer_dim = 3
    output_dim = 2
    dropout_prob = 0.0
    sequence_length = 5

    learning_rate = 0.000001
    momentum = 0.9
    weight_decay = 1e-6
    loss_fn = nn.MSELoss(reduction="mean")

    batch_losses = []
    train_losses = []
    batch_val_losses = []
    val_losses = []

    # rnn = GRUModel(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob)
    # rnn = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob)
    rnn = RNNModel(input_dim, hidden_dim, layer_dim, output_dim, dropout_prob)

    # optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    optimizer = torch.optim.SGD(
        rnn.parameters(), lr=learning_rate ,weight_decay=weight_decay
    )

    train_batches = list(zip(times, lats, lons, delta_lats, delta_lons, locs))
    validate_batches = list(
        zip(
            validate_times,
            validate_lats,
            validate_lons,
            validate_delta_lats,
            validate_delta_lons,
            validate_locs,
        )
    )
    test_batches = list(
        zip(
            test_times,
            test_lats,
            test_lons,
            test_delta_lats,
            test_delta_lons,
            test_locs,
        )
    )

    rnn = rnn.to(device)
    for epoch in range(num_epochs):
        # train
        train_batches = shuffle(train_batches)
        for i, train_batch in enumerate(tqdm.tqdm(train_batches, desc="train")):
            # shape data
            (
                batch_time,
                batch_lat,
                batch_lon,
                batch_delta_lat,
                batch_delta_lon,
                _,
            ) = train_batch
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
            ).split(sequence_length)
            rnn_input = torch.stack(x[:-1], 0)
            y = torch.cat((batch_delta_lat, batch_delta_lon), 1).split(sequence_length)
            rnn_y = torch.cat([z[0] for z in y], 0).reshape(-1, 2)[1:]

            rnn_input = rnn_input.to(device)
            rnn_y = rnn_y.to(device)

            # run model
            optimizer.zero_grad()
            rnn.train()
            output = rnn(rnn_input)

            loss = loss_fn(rnn_y, output)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

            if i + 1 == len(train_batches):
                print("=======================train===========================")
                print(f"{rnn_y[-1]} | {output[-1]}")

        training_loss = np.mean(batch_losses)
        batch_losses.clear()
        train_losses.append(training_loss)

        # validate
        with torch.no_grad():
            for i, validate_batch in enumerate(
                tqdm.tqdm(validate_batches, desc="validate")
            ):
                (
                    batch_time,
                    batch_lat,
                    batch_lon,
                    batch_delta_lat,
                    batch_delta_lon,
                    _,
                ) = validate_batch
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
                ).split(sequence_length)
                rnn_input = torch.stack(x[:-1], 0)
                y = torch.cat((batch_delta_lat, batch_delta_lon), 1).split(
                    sequence_length
                )
                rnn_y = torch.cat([z[0] for z in y], 0).reshape(-1, 2)[1:]

                rnn_input = rnn_input.to(device)
                rnn_y = rnn_y.to(device)

                rnn.eval()
                output = rnn(rnn_input)

                loss = loss_fn(rnn_y, output)

                batch_val_losses.append(loss.cpu())
                if i + 1 == len(train_batches):
                    print("=======================validate===========================")
                    print(f"{rnn_y[-1]} | {output[-1]}")

            validation_loss = np.mean(batch_val_losses)
            batch_val_losses.clear()
            val_losses.append(validation_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], TrainingLoss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
        )

    # plot loss
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title("Losses")
    plt.savefig("loss.png")
    # plt.show()
    # plt.close()

    with torch.no_grad():
        batch_test_losses = []
        test_losses = []
        predictions = []
        values = []
        ref = []
        initial_location = []

        for i, test_batche in enumerate(tqdm.tqdm(test_batches, desc="test")):
            (
                batch_time,
                batch_lat,
                batch_lon,
                batch_delta_lat,
                batch_delta_lon,
                _,
            ) = test_batche
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
            ).split(sequence_length)
            rnn_input = torch.stack(x[:-1], 0)
            y = torch.cat((batch_delta_lat, batch_delta_lon), 1).split(sequence_length)
            rnn_y = torch.cat([z[0] for z in y], 0).reshape(-1, 2)[1:]

            rnn_input = rnn_input.to(device)
            rnn_y = rnn_y.to(device)

            rnn.eval()
            output = rnn(rnn_input)
            loss = loss_fn(rnn_y, output)

            rnn_y = rnn_y.cpu()

            ref.append(rnn_y.cpu())
            predictions.append(output.cpu())
            initial_location.append(rnn_input[:, -1, 1:3].cpu())

            batch_test_losses.append(loss.cpu())
            if i + 1 == len(train_batches):
                print("=======================validate===========================")
                print(f"{rnn_y[-1]} | {output[-1]}")

        test_loss = np.mean(batch_test_losses)

    ref = torch.cat(ref, 0).reshape(-1, 2)
    predictions = torch.cat(predictions, 0).reshape(-1, 2)
    initial_location = torch.cat(initial_location, 0).reshape(-1, 2)
    # print(ref)
    # print(predictions)
    print(f"test_loss: {test_loss}")

    min_lat = -0.04466687620000087
    min_lon = -0.04463771669999517

    max_lat = 0.04456799329999939
    max_lon = 0.04439055920000001

    distance_error = torch.sqrt(
        (
            (ref[:, 0] * (max_lat - min_lat) + min_lat) * 111
            - (predictions[:, 0] * (max_lat - min_lat) + min_lat) * 111
        )
        ** 2
        + (
            (ref[:, 1] * (max_lon - min_lon) + min_lon) * 111
            - (predictions[:, 1] * (max_lon - min_lon) + min_lon) * 111
        )
        ** 2
    )

    # predictions = torch.sqrt((predictions[:,0] * 111) ** 2 + (predictions[:,1] * 111) ** 2)
    # ref = torch.sqrt(
    #     ((ref[:, 0] * (max_lat - min_lat) + min_lat) * 111) ** 2
    #     + ((ref[:, 1] * (max_lon - min_lon) + min_lon) * 111) ** 2
    # )
    # predictions = torch.sqrt(
    #     ((predictions[:, 0] * (max_lat - min_lat) + min_lat) * 111) ** 2
    #     + ((predictions[:, 1] * (max_lon - min_lon) + min_lon) * 111) ** 2
    # )
    del train_batches
    del validate_batches
    del test_batches
    del rnn

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
    print("ref done")

    a = [True if b[i] in np.argpartition(np.sum((keys - node.tolist())**2, axis=1), 1)[:1] else False for i, node in enumerate(tqdm.tqdm(predictions, desc="predictions_1"))]
    print("Prediction done")
    acc = 0 
    for i, _ in enumerate(tqdm.tqdm(range(len(a)), desc = "acc")):
        if a[i]:
            acc += 1
    print(f"val: {acc}")
    print(f"size: {len(ref)}")
    print(f"acc_1: {acc/len(ref)}")

    a = [True if b[i] in np.argpartition(np.sum((keys - node.tolist())**2, axis=1), 5)[:5] else False for i, node in enumerate(tqdm.tqdm(predictions, desc="predictions_5"))]
    print("Prediction done")
    acc = 0 
    for i, _ in enumerate(tqdm.tqdm(range(len(a)), desc = "acc")):
        if a[i]:
            acc += 1
    print(f"val: {acc}")
    print(f"size: {len(ref)}")
    print(f"acc_5: {acc/len(ref)}")

    a = [True if b[i] in np.argpartition(np.sum((keys - node.tolist())**2, axis=1), 10)[:10] else False for i, node in enumerate(tqdm.tqdm(predictions, desc="predictions_10"))]
    print("Prediction done")
    acc = 0 
    for i, _ in enumerate(tqdm.tqdm(range(len(a)), desc = "acc")):
        if a[i]:
            acc += 1
    print(f"val: {acc}")
    print(f"size: {len(ref)}")
    print(f"acc_10: {acc/len(ref)}")
    

    # acc = 0 
    # for i, _ in enumerate(tqdm.tqdm(range(len(a)), desc = "acc")):
    #     if b[i] in a[i][:5]:
    #         acc += 1
    # print(f"val: {acc}")
    # print(f"size: {len(ref)}")
    # print(f"acc_5: {acc/len(ref)}")

    # acc = 0 
    # for i, _ in enumerate(tqdm.tqdm(range(len(a)), desc = "acc")):
    #     if b[i] in a[i][:10]:
    #         acc += 1
    # print(f"val: {acc}")
    # print(f"size: {len(ref)}")
    # print(f"acc_10: {acc/len(ref)}")

    # print(ref)
    # print(predictions)
    # print(location_dict)
    print(f"distance error: {distance_error}")
    print(f"mean distance error: {torch.mean(distance_error)}")



if __name__ == "__main__":
    main()


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

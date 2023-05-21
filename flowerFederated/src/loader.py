import math

import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
import tqdm
import itertools
import torch
from sklearn.utils import shuffle


class Loader:
    def __init__(self, filename):
        self.filename = filename
        self.min_checkins = 40
        self.max_users = 100

    def read_data(self):
        f = open(self.filename, "r")
        entries = f.readlines()
        num_users = 0

        users = []
        times = []
        lats = []
        lons = []
        delta_lats = []
        delta_lons = []
        delta = []
        bearing = []
        locs = []

        validate_times = []
        validate_lats = []
        validate_lons = []
        validate_delta_lats = []
        validate_delta_lons = []
        validate_delta = []
        validate_bearing = []
        validate_locs = []

        test_times = []
        test_lats = []
        test_lons = []
        test_delta_lats = []
        test_delta_lons = []
        test_delta = []
        test_bearing = []
        test_locs = []

        user_time = []
        user_lat = []
        user_lon = []
        user_delta_lats = []
        user_delta_lons = []
        user_delta = []
        user_bearing = []
        user_loc = []

        location_dict = {}

        checkins = 0
        tokens = entries[0].split("\t")
        prev_user = float(tokens[0])
        prev_time = (
            float(
                (
                    datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")
                    - datetime(2009, 1, 1)
                ).total_seconds()
            )
            / 3600
        )
        prev_lat = float(tokens[2])
        prev_lon = float(tokens[3])

        for i, line in enumerate(tqdm.tqdm(entries, desc="read data")):
            tokens = line.split("\t")
            user = float(tokens[0])
            time = (
                float(
                    (
                        datetime.strptime(tokens[1], "%Y-%m-%dT%H:%M:%SZ")
                        - datetime(2009, 1, 1)
                    ).total_seconds()
                )
                / 3600
            )
            lat = float(tokens[2])
            lon = float(tokens[3])
            loc = float(tokens[4])


            if prev_user == user and (
                (lat >= -90 and lat <= 90 and lon >= -180 and lon <= 180)
                and (prev_time - time <= 48)
                and ((prev_lat - lat) ** 2 +  (prev_lon - lon) ** 2 <= 0.002) 
            ):
                if lat >= -90 and lat <= 90 and lon >= -180 and lon <= 180:
                    location_dict[(lat, lon)] = loc
                    checkins += 1
                    user_time.insert(0, prev_time - time)
                    user_lat.insert(0, lat)
                    user_lon.insert(0, lon)
                    user_delta_lats.insert(0, prev_lat - lat)
                    user_delta_lons.insert(0, prev_lon - lon)
                    coords_1 = (prev_lat, prev_lon)
                    coords_2 = (lat, lon)
                    user_loc.insert(0, loc)
                    prev_time = time
                    prev_lat = lat
                    prev_lon = lon
            else:
                if checkins >= self.min_checkins:
                    # num_users += 1
                    # if num_users >= self.max_users:
                    #     break
                    split_validation = int(len(user_time) * 0.85)
                    split_test = int(len(user_time) * 1)

                    users.append(prev_user)
                    user_time = np.roll(user_time, 1)
                    user_delta_lats = np.roll(user_delta_lats, 1)
                    user_delta_lons = np.roll(user_delta_lons, 1)
                    times.append(user_time[:split_validation])
                    lats.append(user_lat[:split_validation])
                    lons.append(user_lon[:split_validation])
                    delta_lats.append(user_delta_lats[:split_validation])
                    delta_lons.append(user_delta_lons[:split_validation])
                    locs.append(user_loc[:split_validation])

                    validate_times.append(user_time[split_validation:split_test])
                    validate_lats.append(user_lat[split_validation:split_test])
                    validate_lons.append(user_lon[split_validation:split_test])
                    validate_delta_lats.append(
                        user_delta_lats[split_validation:split_test]
                    )
                    validate_delta_lons.append(
                        user_delta_lons[split_validation:split_test]
                    )
                    validate_locs.append(user_loc[split_validation:split_test])

                    test_times.append(user_time[split_validation:split_test])
                    test_lats.append(user_lat[split_validation:split_test])
                    test_lons.append(user_lon[split_validation:split_test])
                    test_delta_lats.append(user_delta_lats[split_validation:split_test])
                    test_delta_lons.append(user_delta_lons[split_validation:split_test])
                    test_locs.append(user_loc[split_validation:split_test])

                if lat >= -90 and lat <= 90 and lon >= -180 and lon <= 180:
                    prev_user = user
                    checkins = 1

                    user_time = []
                    user_lat = []
                    user_lon = []
                    user_loc = []
                    user_delta_lats = []
                    user_delta_lons = []

                    prev_time = time
                    prev_lat = lat
                    prev_lon = lon
                    user_time.insert(0, 0)
                    user_lat.insert(0, lat)
                    user_lon.insert(0, lon)
                    user_delta_lats.insert(0, 0)
                    user_delta_lons.insert(0, 0)
                    user_loc.insert(0, loc)
                    location_dict[(lat, lon)] = loc
                else:
                    prev_user = -1
                    checkins = -1
        if checkins >= self.min_checkins:
            split_validation = int(len(user_time) * 0.85)
            split_test = int(len(user_time) * 1)

            users.append(prev_user)
            user_time = np.roll(user_time, 1)
            user_delta_lats = np.roll(user_delta_lats, 1)
            user_delta_lons = np.roll(user_delta_lons, 1)
            times.append(user_time[:split_validation])
            lats.append(user_lat[:split_validation])
            lons.append(user_lon[:split_validation])
            delta_lats.append(user_delta_lats[:split_validation])
            delta_lons.append(user_delta_lons[:split_validation])
            locs.append(user_loc[:split_validation])

            validate_times.append(user_time[split_validation:split_test])
            validate_lats.append(user_lat[split_validation:split_test])
            validate_lons.append(user_lon[split_validation:split_test])
            validate_delta_lats.append(user_delta_lats[split_validation:split_test])
            validate_delta_lons.append(user_delta_lons[split_validation:split_test])
            validate_locs.append(user_loc[split_validation:split_test])

            test_times.append(user_time[split_validation:split_test])
            test_lats.append(user_lat[split_validation:split_test])
            test_lons.append(user_lon[split_validation:split_test])
            test_delta_lats.append(user_delta_lats[split_validation:split_test])
            test_delta_lons.append(user_delta_lons[split_validation:split_test])
            test_locs.append(user_loc[split_validation:split_test])
        print(f"Collected entries for {len(users)} users")
        min_time = min(
            min(min(x) for x in times),
            min(min(x) for x in validate_times),
            min(min(x) for x in test_times),
        )
        min_lat = min(
            min(min(x) for x in lats),
            min(min(x) for x in validate_lats),
            min(min(x) for x in test_lats),
        )
        min_lon = min(
            min(min(x) for x in lons),
            min(min(x) for x in validate_lons),
            min(min(x) for x in test_lons),
        )
        print("==========min==========")
        print(min_time)
        print(min_lat)
        print(min_lon)
        print("==========max==========")
        max_time = max(
            max(max(x) for x in times),
            max(max(x) for x in validate_times),
            max(max(x) for x in test_times),
        )
        max_lat = max(
            max(max(x) for x in lats),
            max(max(x) for x in validate_lats),
            max(max(x) for x in test_lats),
        )
        max_lon = max(
            max(max(x) for x in lons),
            max(max(x) for x in validate_lons),
            max(max(x) for x in test_lons),
        )
        print(max_time)
        print(max_lat)
        print(max_lon)
  
        lats = list(map(lambda x: [float(y - min_lat) / float(max_lat - min_lat) for y in x], lats))
        validate_lats = list(map(lambda x: [float(y - min_lat) / float(max_lat - min_lat)  for y in x], validate_lats))
        test_lats = list(map(lambda x: [float(y - min_lat) / float(max_lat - min_lat)  for y in x], test_lats))

        lons = list(map(lambda x: [float(y - min_lon) / float(max_lon - min_lon)  for y in x], lons))
        validate_lons = list(map(lambda x: [float(y - min_lon) / float(max_lon - min_lon)   for y in x], validate_lons))
        test_lons = list(map(lambda x: [float(y - min_lon) / float(max_lon - min_lon)   for y in x], test_lons))

        min_lat = min(
            min(min(x) for x in delta_lats),
            min(min(x) for x in validate_delta_lats),
            min(min(x) for x in test_delta_lats),
        )
        min_lon = min(
            min(min(x) for x in delta_lons),
            min(min(x) for x in validate_delta_lons),
            min(min(x) for x in test_delta_lons),
        )
        print("==========min==========")
        print(min_lat)
        print(min_lon)
        print("==========max==========")
        max_lat = max(
            max(max(x) for x in delta_lats),
            max(max(x) for x in validate_delta_lats),
            max(max(x) for x in test_delta_lats),
        )
        max_lon = max(
            max(max(x) for x in delta_lons),
            max(max(x) for x in validate_delta_lons),
            max(max(x) for x in test_delta_lons),
        )
        print(max_lat)
        print(max_lon)
        print("==========deviation==========")

        total_lat = torch.FloatTensor(
            list(itertools.chain.from_iterable(delta_lats))
            + list(itertools.chain.from_iterable(validate_delta_lats))
            + list(itertools.chain.from_iterable(test_delta_lats))
        ).reshape(1, -1)
        mean_lat = torch.mean(total_lat)
        deviation_lat = math.sqrt(torch.sum((total_lat - mean_lat) ** 2) / len(total_lat[0]))

        total_lon = torch.FloatTensor(
            list(itertools.chain.from_iterable(delta_lons))
            + list(itertools.chain.from_iterable(validate_delta_lons))
            + list(itertools.chain.from_iterable(test_delta_lons))
        ).reshape(1, -1)
        mean_lon = torch.mean(total_lon)
        deviation_lon = math.sqrt(torch.sum((total_lon - mean_lon) ** 2) / len(total_lon[0]))
        

        print(deviation_lat)
        print(deviation_lon)


        delta_lats = list(
            map(
                lambda x: [
                    float(y - min_lat) / float(max_lat - min_lat) for y in x
                ],
                delta_lats,
            )
        )
        validate_delta_lats = list(
            map(
                lambda x: [
                    float(y - min_lat) / float(max_lat - min_lat) for y in x
                ],
                validate_delta_lats,
            )
        )
        test_delta_lats = list(
            map(
                lambda x: [
                    float(y - min_lat) / float(max_lat - min_lat) for y in x
                ],
                test_delta_lats,
            )
        )

        delta_lons = list(
            map(
                lambda x: [
                    float(y - min_lon) / float(max_lon - min_lon) for y in x
                ],
                delta_lons,
            )
        )
        validate_delta_lons = list(
            map(
                lambda x: [
                    float(y - min_lon) / float(max_lon - min_lon) for y in x
                ],
                validate_delta_lons,
            )
        )
        test_delta_lons = list(
            map(
                lambda x: [
                    float(y - min_lon) / float(max_lon - min_lon) for y in x
                ],
                test_delta_lons,
            )
        )

        times = list(map(lambda x: [float(y - min_time) / float(max_time - min_time)  for y in x], times))
        validate_times = list(map(lambda x: [float(y - min_time) / float(max_time - min_time) for y in x], validate_times))
        test_times = list(map(lambda x: [float(y - min_time) / float(max_time - min_time) for y in x], test_times))

        print("==========preprocess_done==========")

        return (
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
        )

    def get_dataset(self):
        dataset = Data(self.users, self.destinations)

        return DataLoader(dataset)

    def split(self, fullList):
        n = len(fullList)
        return fullList[:int(0.7 * n)], fullList[int(0.7 * n):int(0.85 * n)], fullList[int(0.85 * n):]
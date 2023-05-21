import ast
from dotenv import dotenv_values
from loader import Loader
import os
import random
from server import NUM_CLIENTS

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

    f = open(f'location_dict.txt', 'w')
    f.write(str(location_dict))
    f.close()


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
    
    dir = 'user_data'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    dir = 'validate_data'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))

    dir = 'test_data'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    

    train_len = len(train_batches)
    validate_len = len(validate_batches)
    test_len = len(test_batches)
    used = 0
    for id in range(NUM_CLIENTS):
        id = int(id)
        start = used
        if id == NUM_CLIENTS - 1:
            end = 100
        else:
            end = start
            while end - start > 20 or end - start < 5:
                end = random.randint(start, 100 - 5 * (NUM_CLIENTS - id))
        print (start, end)
        used = end
        start /= 100
        end /= 100

        f = open(f'user_data/user_{id}.txt', 'a')
        f.write(str(train_batches[int(train_len * start) : int(train_len * end)]))
        f.close()

        f = open(f'validate_data/user_{id}.txt', 'a')
        f.write(str(validate_batches[int(validate_len * start) : int(validate_len * end)]))
        f.close()

        f = open(f'test_data/user_{id}.txt', 'a')
        f.write(str(test_batches[int(test_len * start) : int(test_len * end)]))
        f.close()
    
if __name__ == "__main__":
    main()
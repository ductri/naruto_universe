import logging
import pandas as pd

from naruto_skills import dataset_v2

if __name__ == '__main__':
    """
    - Before: 4g -> 6g
    - After: 3.8g -> 3.9g
    """

    logging.basicConfig(level=logging.INFO)
    path_to_csv = '/home/ductri/code/dataset/sentiment/temp.csv'

    # Step 1: Create a generator to generate each element
    def my_generator():
        chunks = pd.read_csv(path_to_csv, chunksize=1000, header=None)
        for df in chunks:
            for row in df[0]:
                yield row


    data_generator = my_generator()
    dataset_v2.split_data_and_dump(data_generator,
                                   dir_data='/home/ductri/code/dataset/temp',
                                   eval_size=50,
                                   test_size=50, chunk_size=10)

    # Reload dumped dataset
    train_dataset = dataset_v2.NarutoDataset(path_to_meta_data='/home/ductri/code/dataset/temp/train_meta_data.json')
    my_iterator = train_dataset.get_data_iterator(batch_size=10)
    real_data = next(my_iterator)
    for item in real_data:
        print(item)

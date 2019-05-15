import unittest
import logging

from model.data_for_train import dataset


class TestDataset(unittest.TestCase):

    def test_success(self):
        X = list(range(100))
        y = [0] * 50 + [1]*50
        my_dataset = dataset.Dataset.create_from_entire_data(X, y, train_prop=0.8, eval_prop=0.2, test_prop=0.0)
        train_iterator = my_dataset.data_train.get_data_iterator(batch_size=10, num_epochs=2)
        eval_iterator = my_dataset.data_eval.get_data_iterator(batch_size=10, num_epochs=2)

        counter = 0
        for X_train, y_train in train_iterator:
            self.assertEqual(len(X_train), 10)
            self.assertEqual(len(y_train), 10)
            counter += 1

        self.assertEqual(counter, 16)
        counter = 0
        for X_test, y_test in eval_iterator:
            self.assertEqual(len(X_test), 10)
            self.assertEqual(len(y_test), 10)
            counter += 1

        self.assertEqual(counter, 4)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    unittest.main()

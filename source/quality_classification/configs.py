import os
current_dir = os.path.dirname(os.path.abspath(__file__))

class BasicConfig:
    def __init__(self):
        self._task = 'quality_classification'
        self._train_dataset_path = 'dataset/train_image_set_fixed.hdf5'
        self._test_dataset_path = 'dataset/test_image_set_fixed.hdf5'
        self._checkpoint_path = os.path.join(current_dir, 'quality_check_checkpoint/')
        self._results_path = 'quality_check_results/'
        self._task_id = 'fixeddata'
        self._model = 'ResNet'
        self._random_seed_list = [41, 42, 43, 44, 45]
        self._batch_size = 4
        self._epochs = 10
        self._learning_rate = 1e-3
        self._loss_weight_good = 1
        self._loss_weight_bad = 2

    # Getters for all attributes
    @property
    def task(self):
        return self._task

    @property
    def train_dataset_path(self):
        return self._train_dataset_path

    @property
    def test_dataset_path(self):
        return self._test_dataset_path

    @property
    def checkpoint_path(self):
        return self._checkpoint_path

    @property
    def results_path(self):
        return self._results_path

    @property
    def task_id(self):
        return self._task_id

    @property
    def model(self):
        return self._model

    @property
    def random_seed_list(self):
        return self._random_seed_list

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def epochs(self):
        return self._epochs

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def loss_weight_good(self):
        return self._loss_weight_good

    @property
    def loss_weight_bad(self):
        return self._loss_weight_bad

    # Setters for all attributes
    @task.setter
    def task(self, task):
        assert isinstance(task, str), "task must be a string"
        self._task = task

    @train_dataset_path.setter
    def train_dataset_path(self, train_dataset_path):
        assert isinstance(train_dataset_path, str), "train_dataset_path must be a string"
        self._train_dataset_path = train_dataset_path

    @test_dataset_path.setter
    def test_dataset_path(self, test_dataset_path):
        assert isinstance(test_dataset_path, str), "test_dataset_path must be a string"
        self._test_dataset_path = test_dataset_path

    @checkpoint_path.setter
    def checkpoint_path(self, checkpoint_path):
        assert isinstance(checkpoint_path, str), "checkpoint_path must be a string"
        self._checkpoint_path = checkpoint_path

    @results_path.setter
    def results_path(self, results_path):
        assert isinstance(results_path, str), "results_path must be a string"
        self._results_path = results_path

    @task_id.setter
    def task_id(self, task_id):
        assert isinstance(task_id, str), "task_id must be a string"
        self._task_id = task_id

    @model.setter
    def model(self, model):
        assert isinstance(model, str), "model must be a string"
        assert model in ["ResNet", "CNN"], "model must be one of ResNet, CNN"
        self._model = model

    @random_seed_list.setter
    def random_seed_list(self, random_seed_list):
        assert isinstance(random_seed_list, list), "random_seed_list must be a list"
        assert all(
            isinstance(seed, int) for seed in random_seed_list), "all elements in random_seed_list must be integers"
        self._random_seed_list = random_seed_list

    @batch_size.setter
    def batch_size(self, batch_size):
        assert isinstance(batch_size, int), "batch_size must be an integer"
        assert batch_size > 0, "batch_size must be greater than 0"
        self._batch_size = batch_size

    @epochs.setter
    def epochs(self, epochs):
        assert isinstance(epochs, int), "epochs must be an integer"
        assert epochs > 0, "epochs must be greater than 0"
        self._epochs = epochs

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        assert isinstance(learning_rate, float), "learning_rate must be a float"
        assert learning_rate > 0, "learning_rate must be greater than 0"
        self._learning_rate = learning_rate

    @loss_weight_good.setter
    def loss_weight_good(self, loss_weight_good):
        assert isinstance(loss_weight_good, float) or isinstance(loss_weight_good,
                                                                 int), "loss_weight_good must be a float or int"
        assert loss_weight_good > 0, "loss_weight_good must be greater than 0"
        self._loss_weight_good = loss_weight_good

    @loss_weight_bad.setter
    def loss_weight_bad(self, loss_weight_bad):
        assert isinstance(loss_weight_bad, float) or isinstance(loss_weight_bad,
                                                                int), "loss_weight_bad must be a float or int"
        assert loss_weight_bad > 0, "loss_weight_bad must be greater than 0"
        self._loss_weight_bad = loss_weight_bad

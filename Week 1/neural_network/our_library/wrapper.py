from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
from .tools import binary_accuracy, multi_class_f1


class Wrapper:

    def __init__(self, model, loss, learning_rate=0.001, multi_class=False, custom_metric_function=None):

        self.model = model
        self.loss = loss
        self.learning_rate = learning_rate

        self.multi_class = multi_class

        self.metric_function = custom_metric_function or (binary_accuracy if not self.multi_class else multi_class_f1)

        self.losses = []
        self.losses_per_epoch = []

        self.accuracies = []
        self.accuracies_per_epoch = []
        self.accuracies_per_epoch_test = []

    def train(self, train_loader, test_loader, epochs=15):

        progress_bar = None

        try:

            for n_epoch in range(epochs):

                epoch_losses = []
                epoch_accuracies = []
                epoch_accuracies_test = []

                progress_bar = tqdm(total=train_loader.dataset.data.shape[0], desc='Epoch {}'.format(n_epoch + 1))

                for batch in train_loader:

                    x, y = self.batch_processing(batch)

                    loss_batch, prediction = self.train_batch(x, y)

                    self.losses.append(loss_batch)
                    epoch_losses.append(loss_batch)

                    batch_accuracy = self.metric_function(y, prediction)

                    self.accuracies.append(batch_accuracy)
                    epoch_accuracies.append(batch_accuracy)

                    progress_bar.update(x.shape[0])

                    postfix_dict = dict()

                    postfix_dict['loss'] = np.mean(epoch_losses)
                    postfix_dict['metric'] = np.mean(epoch_accuracies)

                    progress_bar.set_postfix(**postfix_dict)

                for batch in test_loader:

                    x, y = self.batch_processing(batch)

                    prediction = self.predict_batch(x)

                    batch_accuracy = self.metric_function(y, prediction)

                    epoch_accuracies_test.append(np.mean(batch_accuracy))

                self.losses_per_epoch.append(np.mean(epoch_losses))
                self.accuracies_per_epoch.append(np.mean(epoch_accuracies))
                self.accuracies_per_epoch_test.append(np.mean(epoch_accuracies_test))

                progress_bar.close()

        except KeyboardInterrupt:

            if progress_bar:
                progress_bar.close()

    @staticmethod
    def batch_processing(batch):

        x, y = batch

        x = x.view(x.shape[0], -1).numpy()
        y = y.numpy()

        return x, y

    def train_batch(self, x, y):

        raise NotImplementedError

    def predict_batch(self, x):

        raise NotImplementedError

    def plot(self):

        plt.figure(figsize=(16, 12))

        plt.subplot(1, 2, 1)

        plt.grid()
        plt.xlabel('Training step', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.plot(self.losses)

        plt.subplot(1, 2, 2)

        plt.grid()
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Metric', fontsize=12)
        plt.plot(self.accuracies_per_epoch, label='Train')
        plt.plot(self.accuracies_per_epoch_test, label='Test')
        plt.ylim(0, 1)
        plt.legend()

        plt.show()


class MNISTWrapper(Wrapper):

    def __init__(self, model, loss, learning_rate=0.001, multi_class=False):

        # передаем параметры, чтобы отработал __init__ от наследуемого класса
        # можно было бы сделать через **kwargs, но сделал так для наглядности и чтобы были подсказки в юпитере

        super().__init__(model=model, loss=loss, learning_rate=learning_rate, multi_class=multi_class)

    def train_batch(self, x, y):
        """
        Нужно реализовать одну итерацию обучения модели:
        1. Рассчет forward
        2. Рассчет функции потерь
        3. Рассчет backward от функции потерь
        4. Рассчет backward по модели
        5. Обновление весов
        :param x: входные данные np.array with shape (batch_size, n_features)
        :param y: предсказания np.array with shape (batch_size, n_classes)
        :return:
        loss_batch - значение функции потерь, просто скаляр
        prediction - матрица предсказаний вашей модели

        напомню важные штуки, которые знает наш класс:
        self.model
        self.loss
        self.learning_rate
        """

    def predict_batch(self, x):
        """
        Предсказание (aka inference) вашей модели:
        1. Рассчет forward
        :param x: входные данные np.array with shape (batch_size, n_features)
        :return: prediction - матрица предсказаний вашей модели
        """

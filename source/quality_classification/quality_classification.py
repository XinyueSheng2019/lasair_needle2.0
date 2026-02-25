from .configs import BasicConfig
from .models import ResNet, CNN
from .earlystop import EarlyStopping, adjust_learning_rate
from .preprocess import preprocessing, zscale, image_normal

import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from sklearn.metrics import roc_curve, auc
import tensorflow as tf



class QualityClassification:
    def __init__(self, verbose=True):
        self.args = BasicConfig()
        self._seed_for_model()
        self._model_list = None
        self._model_list_setting = None
        self._loss_weights = None
        self._label_dict = {'Good': 1, 'Bad': 0}
        self._verbose = verbose

        if self._verbose:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print("Running on GPU")
                for gpu in gpus:
                    print(f"device: {gpu.name}, type: {gpu.device_type}")
            else:
                print("No GPU detected, running on CPU")

    def _print_properties(self, obj):
        cls = type(obj)

        properties = {}
        for name in dir(cls):
            attr = getattr(cls, name)
            if isinstance(attr, property):
                try:
                    value = getattr(obj, name)
                    properties[name] = value
                except Exception as e:
                    properties[name] = f"<Error: {str(e)}>"

        for name, value in properties.items():
            print(f"  - {name}: {value}")

    def _seed_for_model(self, seed=42):
        tf.random.set_seed(seed)
        np.random.seed(seed)

    def _build_model(self):
        if self.args.model == 'ResNet':
            model = ResNet()
        elif self.args.model == 'CNN':
            model = CNN(1)
        else:
            raise NotImplementedError
        return model

    def _build_model_list(self):
        model_list = []
        input_shape = (60, 60, 1)
        for model_i in range(len(self.args.random_seed_list)):
            random_seed = self.args.random_seed_list[model_i]
            model_setting = (f'{self.args.task}_'
                             f'{self.args.task_id}_'
                             f'{self.args.model}_'
                             f'seed{random_seed}_'
                             f'bs{self.args.batch_size}_'
                             f'ep{self.args.epochs}')
            # Build model
            model = self._build_model()
            model.build(input_shape=(None, *input_shape))  # Initialize variables
            # Construct path
            checkpoint_dir = os.path.join(self.args.checkpoint_path, model_setting)
            model_path = os.path.join(checkpoint_dir, "checkpoint.ckpt")
            # Load weights only if checkpoint exists
            if os.path.exists(model_path + ".index"):
                try:
                    #model.load_weights(model_path)
                    checkpoint = tf.train.Checkpoint(model)
                    # Restore the checkpointed values to the `model` object.
                    checkpoint.restore(model_path).expect_partial()
                    if self._verbose:
                        print(f"Loaded weights for seed {random_seed}")
                except Exception as e:
                    print(f"Failed to load {model_path}: {str(e)}")
                    model = None
            else:
                print(model_path + ".index")
                print(f"No checkpoint for seed {random_seed}")
                model = None
            model_list.append(model)
        return model_list

    def _process_dataset_to_int_64(self, x, y):
        return x, tf.cast(y, tf.int64)

    def _dataloader(self, random_seed, flag='train'):
        if flag == 'train':
            dataset_path = self.args.train_dataset_path
            train_set_ratio = 0.8
            train_imageset, train_labels, test_imageset, test_labels = preprocessing(
                dataset_path, train_set_ratio=train_set_ratio, random_seed=random_seed
            )
            train_samples = len(train_imageset)
            train_dataset = tf.data.Dataset.from_tensor_slices((train_imageset, train_labels))
            train_dataset = train_dataset.batch(self.args.batch_size).prefetch(tf.data.AUTOTUNE)
            train_dataset = train_dataset.map(self._process_dataset_to_int_64)
        elif flag == 'test':
            dataset_path = self.args.test_dataset_path
            train_imageset, train_labels, test_imageset, test_labels = preprocessing(
                dataset_path, train_set_ratio=0, random_seed=random_seed
            )
            train_dataset = None
            train_samples = None
        else:
            raise ValueError
        test_samples = len(test_imageset)
        test_dataset = tf.data.Dataset.from_tensor_slices((test_imageset, test_labels))
        test_dataset = test_dataset.batch(1).prefetch(tf.data.AUTOTUNE)
        test_dataset = test_dataset.map(self._process_dataset_to_int_64)
        criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate)
        return train_dataset, test_dataset, train_samples, test_samples, criterion, optimizer

    def _train_model(self, model_i):
        model = self._build_model()
        random_seed = self.args.random_seed_list[model_i]
        setting = (f'{self.args.task}_'
                   f'{self.args.task_id}_'
                   f'{self.args.model}_'
                   f'seed{random_seed}_'
                   f'bs{self.args.batch_size}_'
                   f'ep{self.args.epochs}')
        train_dataset, test_dataset, train_samples, test_samples, criterion, optimizer = self._dataloader(random_seed, 'train')
        early_stopping = EarlyStopping(patience=3, verbose=True, delta=1e-6)

        print("### Start to train model with random seed {} ###".format(random_seed))

        for epoch in range(self.args.epochs):
            total_loss = 0.0
            correct = 0
            for x_batch, y_batch in train_dataset:
                sample_weights = tf.gather(self._loss_weights, y_batch)
                with tf.GradientTape() as tape:
                    output = model(x_batch, training=True)
                    loss = criterion(y_batch, output, sample_weight=sample_weights)
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                total_loss += loss
                pred_probs = tf.math.exp(output)
                preds = tf.argmax(pred_probs, axis=1)
                correct += tf.reduce_sum(tf.cast(preds == y_batch, tf.float32))

            train_loss = total_loss / train_samples
            accuracy = 100 * correct / train_samples

            print(f'epoch: {epoch + 1} | train loss: {train_loss.numpy()} | accuracy: {accuracy.numpy()}')

            eval_loss, eval_accuracy = self._evaluate_model(model, test_dataset, test_samples, criterion)
            print(f'eval loss: {eval_loss.numpy()}')

            save_path = self.args.checkpoint_path + setting
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            early_stopping(eval_loss, model, save_path)
            if early_stopping.early_stop:
                print("\tEarly stopping")
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        print("### Training model with random seed {} done ###".format(random_seed))
        return model

    def _evaluate_model(self, model, test_dataset, test_samples, criterion):
        total_loss = 0.0
        correct = 0
        for x_batch, y_batch in test_dataset:
            sample_weights = tf.gather(self._loss_weights, y_batch)
            output = model(x_batch, training=False)
            loss = criterion(y_batch, output, sample_weight=sample_weights)
            total_loss += loss
            pred_probs = tf.math.exp(output)
            preds = tf.argmax(pred_probs, axis=1)
            correct += tf.reduce_sum(tf.cast(preds == y_batch, tf.float32))

        eval_loss = total_loss / test_samples
        accuracy = 100 * correct / test_samples
        return eval_loss, accuracy

    def test_model(self, only_averaged_model=True, save_wrong_samples=False):
        # check all models exist, if not, train them
        self.check_models()
        self._model_list_setting = (f'{self.args.task}_'
                                    f'{self.args.task_id}_'
                                    f'{self.args.model}_'
                                    f'seed{self.args.random_seed_list}_'
                                    f'bs{self.args.batch_size}_'
                                    f'ep{self.args.epochs}')
        if not only_averaged_model:
            # test for each single model
            for model_i in range(len(self._model_list)):
                model = self._model_list[model_i]
                random_seed = self.args.random_seed_list[model_i]
                print("### Start to test model with random seed {} ###".format(random_seed))
                train_loader, test_loader, train_samples, test_samples, criterion, optimizer = self._dataloader(42,
                                                                                                                'test')
                total_loss = 0.0
                correct = 0
                all_outputs = []
                all_trues = []

                for (x_batch, y_batch) in test_loader:
                    sample_weights = tf.gather(self._loss_weights, y_batch)
                    output = model(x_batch, training=False)

                    loss = criterion(y_batch, output, sample_weight=sample_weights)
                    total_loss += loss.numpy()

                    pred_probs = tf.math.exp(output)
                    preds = tf.argmax(pred_probs, axis=1)
                    correct += tf.reduce_sum(tf.cast(preds == y_batch, tf.float32)).numpy()

                    all_outputs.append(pred_probs.numpy())
                    all_trues.append(y_batch.numpy())

                eval_loss = total_loss / test_samples
                accuracy = 100 * correct / test_samples
                folder_path = os.path.join(self.args.results_path, self._model_list_setting,
                                           f'model_seed_{random_seed}')
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                np.save(folder_path + 'all_outputs.npy', np.concatenate(all_outputs))
                np.save(folder_path + 'all_trues.npy', np.concatenate(all_trues))
                pd.DataFrame(np.concatenate(all_outputs)).to_csv(
                    os.path.join(folder_path, 'all_outputs.csv'),
                    index=False,
                    header=False
                )
                pd.DataFrame(np.concatenate(all_trues)).to_csv(
                    os.path.join(folder_path, 'all_trues.csv'),
                    index=False,
                    header=False
                )
                print(f'model with seed {random_seed} test accuracy:', accuracy.item())
                print(f'model with seed {random_seed} eval_loss:', eval_loss.item())
                print("### Testing model with random seed {} done ###".format(random_seed))

        # test for the averaged models
        print("### Start to test model averaged ###")
        train_loader, test_loader, train_samples, test_samples, criterion, _ = self._dataloader(42, 'test')
        total_loss = 0.0
        correct = 0
        all_outputs = []
        all_trues = []
        wrong_img_idx = 0
        true_labels = {0: 'bad', 1: 'good'}
        folder_path = os.path.join(self.args.results_path, self._model_list_setting, 'averaged')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        save_wrong_img_path = os.path.join(folder_path, 'wrong_img')
        if not os.path.exists(save_wrong_img_path):
            os.makedirs(save_wrong_img_path)
        for (x_batch, y_batch) in test_loader:
            outputs = []
            sample_weights = tf.gather(self._loss_weights, y_batch)
            for model_i in range(len(self._model_list)):
                model = self._model_list[model_i]
                output = model(x_batch, training=False)
                outputs.append(output)
            outputs = tf.stack(outputs, axis=0)
            outputs_averaged = tf.reduce_mean(outputs, axis=0)

            loss = criterion(y_batch, outputs_averaged, sample_weight=sample_weights)
            total_loss += loss.numpy()

            pred_probs = tf.math.exp(outputs_averaged)
            preds = tf.argmax(pred_probs, axis=1)
            correct += tf.reduce_sum(tf.cast(preds == y_batch, tf.float32)).numpy()

            if save_wrong_samples:
                wrong_img_idx += 1
                if not preds == y_batch:
                    img = tf.squeeze(x_batch)
                    fig, ax = plt.subplots()
                    ax.imshow(img, cmap='Blues')
                    pred_probs_value = pred_probs.numpy()
                    preds_value = preds.numpy()
                    y_batch_value = y_batch.numpy()
                    ax.set_title(
                        f'prob: {pred_probs_value[0]}, '
                        f'pred {true_labels[preds_value[0]]}, '
                        f'true {true_labels[y_batch_value[0]]}'
                    )
                    ax.set_axis_off()
                    plt.savefig(save_wrong_img_path + f'/{wrong_img_idx}.png')
                    plt.close()

            all_outputs.append(pred_probs.numpy())
            all_trues.append(y_batch.numpy())

        eval_loss = total_loss / test_samples
        accuracy = 100 * correct / test_samples
        np.save(folder_path + 'all_outputs.npy', np.concatenate(all_outputs))
        np.save(folder_path + 'all_trues.npy', np.concatenate(all_trues))
        pd.DataFrame(np.concatenate(all_outputs)).to_csv(
            os.path.join(folder_path, 'all_outputs.csv'),
            index=False,
            header=False
        )
        pd.DataFrame(np.concatenate(all_trues)).to_csv(
            os.path.join(folder_path, 'all_trues.csv'),
            index=False,
            header=False
        )

        print(f'averaged model with seed list {self.args.random_seed_list} test accuracy:', accuracy.item())
        print(f'averaged model with seed list {self.args.random_seed_list} eval_loss:', eval_loss.item())
        print("### Testing model averaged done ###")

    def check_models(self):
        if self._verbose:
            print("args in experiment:")
            self._print_properties(self.args)
        self._model_list = self._build_model_list()
        self._loss_weights = tf.constant([self.args.loss_weight_good, self.args.loss_weight_bad], dtype=tf.float32)
        # check all models have been trained
        for model_i in range(len(self._model_list)):
            if self._model_list[model_i] is None:
                self._model_list[model_i] = self._train_model(model_i)

    def run(self, target):
        """
        Args:
            target: target image, numpy_array (60, 60)
        Return:
            good_prob (float): prediction probability of good sample in the range [0, 1]
        """
        # check all models exist, if not, train them
        self.check_models()
        self._model_list_setting = (f'{self.args.task}_'
                                    f'{self.args.task_id}_'
                                    f'{self.args.model}_'
                                    f'seed{self.args.random_seed_list}_'
                                    f'bs{self.args.batch_size}_'
                                    f'ep{self.args.epochs}')
        model_training_list = []
        for model_i in range(len(self._model_list)):
            if self._model_list[model_i] is None:
                model_training_list.append(self._model_list[model_i])
        if model_training_list:
            lst = [self.args.random_seed_list[model_i] for model_i in model_training_list]
            raise ValueError(f"model with random seed list {lst} need to be trained!")

        # quality check target image
        # image preprocess
        image = image_normal(zscale(target)).reshape(1, 60, 60, 1)
        # predict
        outputs = []
        for model_i in range(len(self._model_list)):
            model = self._model_list[model_i]
            output = model(image, training=False)
            outputs.append(output)
        outputs = tf.stack(outputs, axis=0)
        outputs_averaged = tf.reduce_mean(outputs, axis=0)
        pred_probs = tf.math.exp(outputs_averaged)
        pred_probs = pred_probs.numpy()
        good_prob = pred_probs[0][1]
        
        return good_prob

    def plot_confusion_matrix(self, model='all'):
        """
        Plots the Confusion Matrix.

        Args:
            model (string): 'all', 'averaged' or seed'{seed_number}'
        """
        model_list = []
        if model == 'all':
            for i in self.args.random_seed_list:
                model_list.append(str(i))
            model_list.append('averaged')
        else:
            model_list.append(model)

        for model_name in model_list:
            if model_name == 'averaged':
                setting = (f'{self.args.task}_'
                           f'{self.args.task_id}_'
                           f'{self.args.model}_'
                           f'seed_averaged_'
                           f'bs{self.args.batch_size}_'
                           f'ep{self.args.epochs}')
                folder_path = self.args.results_path + self._model_list_setting + '/' + f'averaged' + '/'
            else:
                random_seed = int(model_name)
                setting = (f'{self.args.task}_'
                           f'{self.args.task_id}_'
                           f'{self.args.model}_'
                           f'seed{random_seed}_'
                           f'bs{self.args.batch_size}_'
                           f'ep{self.args.epochs}')
                folder_path = self.args.results_path + self._model_list_setting + '/' + f'model_seed_{random_seed}' + '/'

            predictions = np.load(folder_path + 'all_outputs.npy')
            y_pred = np.argmax(predictions, axis=-1)
            y_true = np.load(folder_path + 'all_trues.npy')
            cm = confusion_matrix(y_true, y_pred)

            p_cm = []
            for i in cm:
                p_cm.append(np.round(i / np.sum(i), 3))

            # Get Class Labels
            labels = self._label_dict.keys()
            class_names = labels

            # Plot confusion matrix in a beautiful manner
            fig = plt.figure(figsize=(16, 14))
            font_size = 100
            ax = plt.subplot()
            sns.set(font_scale=8)
            sns.heatmap(cm, annot=True, ax=ax, fmt='g', annot_kws={"size": font_size})  # annot=True to annotate cells

            # labels, title and ticks
            ax.set_xlabel('Predicted', fontsize=font_size)
            ax.xaxis.set_label_position('bottom')
            plt.xticks(rotation=0)
            ax.xaxis.set_ticklabels(class_names, fontsize=font_size)
            ax.xaxis.tick_bottom()
            ax.axis('off')
            ax.set_ylabel('True', fontsize=font_size)
            ax.yaxis.set_ticklabels(class_names, fontsize=font_size)
            ax.tick_params(labelsize=font_size)
            plt.yticks(rotation=90)

            # current_time = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            plt.title('Confusion Matrix', fontsize=font_size)

            plt.savefig(folder_path + 'confusion_matrix' + '.pdf')
            with open(folder_path + 'confusion_matrix' + ".csv", 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                rows = [[setting] + [cm[0][0]] + [cm[0][1]] + [cm[1][0]] + [cm[1][1]]]
                writer.writerows(rows)

            if self._verbose:
                if model_name == 'averaged':
                    print('plot confusion_matrix for model averaged done')
                else:
                    print(f'plot confusion_matrix for model {random_seed} done')

    def __plot_probabilities_distribution(self, all_trues, ax, probabilities, title):
        width = 0.04  # bar width
        bins = np.arange(0, 1.1, 0.1)  # probabilities bins
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # count number of predict samples in the bins
        good_counts = []
        bad_counts = []

        for i in range(len(bins) - 1):
            lower, upper = bins[i], bins[i + 1]
            if i == len(bins) - 2:
                good_mask = (probabilities[:, 1] >= lower) & (probabilities[:, 1] <= upper)
                bad_mask = (probabilities[:, 0] >= lower) & (probabilities[:, 0] <= upper)
            else:
                good_mask = (probabilities[:, 1] >= lower) & (probabilities[:, 1] < upper)
                bad_mask = (probabilities[:, 0] >= lower) & (probabilities[:, 0] < upper)

            # count number of samples
            good_pred = np.sum(all_trues[good_mask] == 1)
            bad_pred = np.sum(all_trues[bad_mask] == 0)

            good_counts.append(good_pred)
            bad_counts.append(bad_pred)
        # draw bars
        bar1 = ax.bar(bin_centers - 0.05 - width / 2, good_counts, width,
                      label='Good', color='#66c2a5', alpha=0.8)
        bar2 = ax.bar(bin_centers - 0.05 + width / 2, bad_counts, width,
                      label='Bad', color='#fc8d62', alpha=0.8)

        # add number of samples labels
        for rect in bar1 + bar2:
            height = rect.get_height()
            if height > 0:
                # y_pos = height - 0.5
                y_pos = height * 0.6  # for log scale
                ax.text(rect.get_x() + rect.get_width() / 2, y_pos,
                        f'{int(height)}',
                        ha='center',
                        va='bottom',
                        fontsize=20,
                        fontweight='bold',
                        alpha=0.8)

        # add a red dash line at 0.5 for correct/incorrect
        ax.axvline(x=0.5 - 0.05, color='red', linestyle='--', linewidth=3, alpha=0.8)

        ax.set_title(title, fontsize=40, pad=20)
        ax.set_xlabel('Predicted Probability', fontsize=35, labelpad=15)
        ax.set_ylabel('Sample Count', fontsize=35, labelpad=15)
        ax.tick_params(axis='both', labelsize=30)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=35, loc='upper left')
        ax.set_xticks(bin_centers)
        ax.set_xticklabels([f'{(i + 1) / 10:.1f}' for i in range(10)])
        ax.set_xlim(-0.05, 1.05)
        # use log scale for y axis
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.1)

    def plot_probabilities_bar(self, model='all'):
        """
        Plots the Probabilities Bar.

        Args:
            model: 'all', 'averaged' or seed'{seed_number}'
        """
        model_list = []
        if model == 'all':
            for i in self.args.random_seed_list:
                model_list.append(str(i))
            model_list.append('averaged')
        else:
            model_list.append(model)

        for model_name in model_list:
            if model_name == 'averaged':
                folder_path = self.args.results_path + self._model_list_setting + '/averaged/'
            else:
                random_seed = int(model_name)
                folder_path = self.args.results_path + self._model_list_setting + f'/model_seed_{random_seed}/'

            all_outputs = np.load(folder_path + 'all_outputs.npy').squeeze()
            all_trues = np.load(folder_path + 'all_trues.npy').squeeze()

            plt.figure(figsize=(20, 10))
            ax = plt.gca()

            self.__plot_probabilities_distribution(
                all_trues=all_trues,
                ax=ax,
                probabilities=all_outputs,
                title='Prediction Probability Distribution'
            )

            plt.tight_layout()
            plt.savefig(folder_path + 'probabilities_bar.pdf')
            plt.close()

            if self._verbose:
                if model_name == 'averaged':
                    print('plot probabilities_bar for averaged model done')
                else:
                    print(f'plot probabilities_bar for model {random_seed} done')

    def plot_roc_curve(self, model='all'):
        """
        Plots the Receiver Operating Characteristic (ROC) curve.
        
        Args:
            model (string): 'all', 'averaged' or seed'{seed_number}'
        """
        model_list = []
        if model == 'all':
            for i in self.args.random_seed_list:
                model_list.append(str(i))
            model_list.append('averaged')
        else:
            model_list.append(model)
        for model_name in model_list:
            if model_name == 'averaged':
                folder_path = self.args.results_path + self._model_list_setting + '/averaged/'
            else:
                random_seed = int(model_name)
                folder_path = self.args.results_path + self._model_list_setting + f'/model_seed_{random_seed}/'

            all_outputs = np.load(folder_path + 'all_outputs.npy').squeeze()
            all_trues = np.load(folder_path + 'all_trues.npy').squeeze()

            # Calculate ROC metrics
            fpr, tfr, thresholds = roc_curve(all_trues, all_outputs[:, 1])
            roc_auc = auc(fpr, tfr)

            # Set style and create figure
            sns.set(style="whitegrid")
            plt.figure(figsize=(10, 8))

            # Plot ROC curve
            plt.plot(fpr, tfr, label=f'AUC = {roc_auc:.2f}', color='darkorange', lw=2)
            plt.plot([0, 1], [0, 1], 'k--')  # Random chance line

            # Customize plot
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve', fontsize=14)
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(True, alpha=0.3)

            plt.savefig(folder_path + 'roc_curve.pdf', bbox_inches='tight')
            plt.close()

            if self._verbose:
                if model_name == 'averaged':
                    print('plot roc_curve for averaged model done')
                else:
                    print(f'plot roc_curve for model {random_seed} done')

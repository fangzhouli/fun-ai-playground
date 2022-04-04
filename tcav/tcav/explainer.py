import tensorflow as tf
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np


class TCAVExplainer:

    def __init__(self):
        pass

    def _get_bottleneck_acts(self, encoder, inputs):
        """
        """
        acts = encoder.predict(inputs)

        return acts

    def _get_cav(
            self,
            acts_concept,
            acts_random,
            lm='svm'):
        """
        """
        if len(acts_concept.shape) == 4:
            # Flatten image data.
            X_pos = tf.reshape(
                acts_concept,
                shape=[acts_concept.shape[0], -1]).numpy()
            X_neg = tf.reshape(
                acts_random,
                shape=[acts_random.shape[0], -1]).numpy()
        else:
            raise NotImplementedError("Only image data is supported.")

        if lm == 'svm':
            lm = SGDClassifier(
                alpha=0.01, max_iter=1000, tol=1e-3)
        elif lm == 'logistic':
            lm = LogisticRegression()
        else:
            raise ValueError("Unknown linear model.")

        # Balancing the classes.
        n_data_per_class = min(len(X_pos), len(X_neg))
        X_pos = X_pos[:n_data_per_class]
        X_neg = X_neg[:n_data_per_class]
        X = np.concatenate([X_pos, X_neg])
        y = np.array(
            [0 for _ in range(len(X_pos))] + [1 for _ in range(len(X_neg))])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, stratify=y)
        lm.fit(X_train, y_train)
        y_pred = lm.predict(X_test)
        print(accuracy_score(y_test, y_pred))

        # Coefficients of the linear model is the CAV. Since binary
        #   classification, coef_ corresponds positive class (1), thus flip
        #   the sign.
        cav = -lm.coef_
        print(cav)

        return cav

    def _get_tcav_score(self, model, inputs_target, cav):
        """
        """
        pass

    def explain(
            self,
            model,
            target,
            bottleneck,
            concept_dataset,
            random_dataset):
        """
        """
        encoder = tf.keras.models.Model(
            inputs=model.input,
            outputs=model.get_layer(bottleneck).output)

        # Get concept activations for each bottleneck.
        inputs = tf.stack([datum[0] for datum in concept_dataset])
        acts_concept = self._get_bottleneck_acts(
            encoder, inputs)

        # Get random activations for each bottleneck.
        inputs = tf.stack([datum[0] for datum in random_dataset])
        acts_random = self._get_bottleneck_acts(
            encoder, inputs)

        cav = [self._get_cav(
            acts_concept,
            acts_random)]


if __name__ == '__main__':
    import PIL
    import tensorflow as tf

    from .utils import load_dataset

    model = tf.keras.applications.mobilenet.MobileNet()
    target = ['zebra']
    concept = 'horse'
    bottleneck = 'conv_pw_12'
    concept_dataset = load_dataset(
        f"/home/lfz/git/fun-ai-playground/tcav/tcav/data/concept/{concept}")

    random_datasets = {}
    for i in range(5):
        random_datasets[i] = load_dataset(
            f"/home/lfz/git/fun-ai-playground/tcav/tcav/data/random/random{i}")

    explainer = TCAVExplainer()
    explainer.explain(
        model=model,
        target=target,
        bottleneck=bottleneck,
        concept_dataset=concept_dataset,
        random_dataset=random_datasets[1])

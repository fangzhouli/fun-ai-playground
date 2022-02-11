import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

logging_config = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "custom": {
            # More format options are available in the official
            # `documentation <https://docs.python.org/3/howto/logging-cookbook.
            # html>`_
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    # Any INFO level msg will be printed to the console
    "handlers": {
        "console": {
            "level": "INFO",
            "formatter": "custom",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "": {  # root logger
            "level": "DEBUG",
        },
        "Client-EnsembleBuilder": {
            "level": "DEBUG",
            "handlers": ["console"],
        },
    },
}


if __name__ == "__main__":
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl = autosklearn.classification.AutoSklearnClassifier(
        n_jobs=-1,
        logging_config=logging_config)
    automl.fit(X_train, y_train)
    y_hat = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

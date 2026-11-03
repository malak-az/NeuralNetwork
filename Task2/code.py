import sys
import io
import time
import itertools
import contextlib

import pandas as pd
import numpy as np

data = pd.read_csv(r"data\penguins.csv")

# added: the task asks for non-repeated samples
data = data.drop_duplicates().reset_index(drop=True)

train_data = []
test_data = []

for species in data["Species"].unique():
    class_data = data[data["Species"] == species]

    train_data.append(class_data.iloc[:30])
    test_data.append(class_data.iloc[30:50])

train_data = pd.concat(train_data)
test_data = pd.concat(test_data)


train_data = train_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

features = [
    "CulmenLength",
    "CulmenDepth",
    "FlipperLength",
    "BodyMass",
    "OriginLocation"
]

# added .copy() to avoid SettingWithCopyWarning
X_train = train_data[features].copy()
y_train = train_data["Species"]

X_test = test_data[features].copy()
y_test = test_data["Species"]

numeric_features = [
    "CulmenLength",
    "CulmenDepth",
    "FlipperLength",
    "BodyMass"
]

meds = X_train[numeric_features].median()

X_train[numeric_features] = X_train[numeric_features].fillna(meds)
X_test[numeric_features] = X_test[numeric_features].fillna(meds)

# added: fill missing OriginLocation with the training mode
location_mode = X_train["OriginLocation"].mode()[0]
X_train["OriginLocation"] = X_train["OriginLocation"].fillna(location_mode)
X_test["OriginLocation"] = X_test["OriginLocation"].fillna(location_mode)

# Encode OriginLocation as one numerical feature

location_map = {
    "Biscoe": 0,
    "Dream": 1,
    "Torgersen": 2
}

X_train["OriginLocation"] = (
    X_train["OriginLocation"].map(location_map)
)

X_test["OriginLocation"] = (
    X_test["OriginLocation"].map(location_map)
)

mean = X_train.mean()
std = X_train.std()

std[std == 0] = 1

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# One-Hot Encoding for target labels

y_train = pd.get_dummies(y_train, dtype=int)
y_test = pd.get_dummies(y_test, dtype=int)

y_test = y_test.reindex(
    columns=y_train.columns,
    fill_value=0
)

# Convert features to NumPy arrays

X_train = X_train.to_numpy(dtype=float)
X_test = X_test.to_numpy(dtype=float)

# Get class names before converting labels
class_names = y_train.columns.tolist()

# Convert one-hot labels to class indices
y_train = np.argmax(y_train.to_numpy(), axis=1)
y_test = np.argmax(y_test.to_numpy(), axis=1)


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(a):
    return 1 - a ** 2


class MLP:

    def __init__(self, input_size, hidden_layers, output_size,
                 learning_rate, activation, use_bias):

        self.learning_rate = learning_rate
        self.use_bias = use_bias
        self.activation_name = activation

        if activation == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        else:
            self.activation = tanh
            self.activation_derivative = tanh_derivative

        self.layers = [input_size] + hidden_layers + [output_size]

        np.random.seed(42)

        self.weights = []
        self.biases = []

        for i in range(len(self.layers) - 1):

            w = np.random.uniform(-0.1, 0.1,(self.layers[i], self.layers[i + 1]))

            b = np.random.uniform(-0.1, 0.1,(1, self.layers[i + 1]))

            self.weights.append(w)
            self.biases.append(b)


    def forward(self, x):

        self.activations = [x.reshape(1, -1)]

        current = self.activations[0]

        for i in range(len(self.weights)):

            z = np.dot(current, self.weights[i])

            if self.use_bias:
                z += self.biases[i]

            current = self.activation(z)

            self.activations.append(current)

        return current


    def backward(self, target):

        output = self.activations[-1]

        error = target - output

        delta = error * self.activation_derivative(output)

        for i in reversed(range(len(self.weights))):

            previous_activation = self.activations[i]
            if i > 0:

                previous_delta = (
                    np.dot(delta, self.weights[i].T)
                    * self.activation_derivative(self.activations[i])
                )

            weight_gradient = np.dot(
                previous_activation.T, delta
            )

            self.weights[i] += (self.learning_rate * weight_gradient)

            if self.use_bias:
                self.biases[i] += self.learning_rate * delta

            if i > 0:
                delta = previous_delta


    def train(self, X, y, epochs):

        # added: history used by the GUI to draw the training curve
        self.history = {"loss": [], "accuracy": []}

        for epoch in range(epochs):

            total_loss = 0.0

            for i in range(len(X)):

                output = self.forward(X[i])

                target = np.zeros((1, len(class_names)))

                if self.activation_name == "sigmoid":
                    target[0, y[i]] = 1

                else:
                    target.fill(-1)
                    target[0, y[i]] = 1

                total_loss += 0.5 * np.sum((target - output) ** 2)

                self.backward(target)

            # added: record loss and accuracy every epoch
            predictions = self.predict(X)

            accuracy = np.mean(predictions == y) * 100

            self.history["loss"].append(total_loss / len(X))
            self.history["accuracy"].append(accuracy)

            if (epoch + 1) % max(1, epochs // 10) == 0:

                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"- Training Accuracy: {accuracy:.2f}%"
                )


    def predict_sample(self, x):

        output = self.forward(x)

        return np.argmax(output)


    def predict(self, X):

        predictions = []

        for sample in X:
            predictions.append(self.predict_sample(sample))

        return np.array(predictions)


def calculate_confusion_matrix(y_true, y_pred, num_classes):

    matrix = np.zeros(
        (num_classes, num_classes),
        dtype=int
    )

    for actual, predicted in zip(y_true, y_pred):

        matrix[actual][predicted] += 1

    return matrix


def run_model(hidden_layers, learning_rate, epochs, activation_name, use_bias):
    """Build, train and test the network (used by the GUI and by main)."""

    model = MLP(
        input_size=X_train.shape[1],
        hidden_layers=hidden_layers,
        output_size=len(class_names),
        learning_rate=learning_rate,
        activation=activation_name,
        use_bias=use_bias
    )

    model.train(X_train, y_train, epochs)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    train_accuracy = np.mean(train_predictions == y_train) * 100
    test_accuracy = np.mean(test_predictions == y_test) * 100

    # Confusion matrix using test predictions
    conf_matrix = calculate_confusion_matrix(
        y_test,
        test_predictions,
        len(class_names)
    )

    confusion_df = pd.DataFrame(
        conf_matrix,
        index=class_names,
        columns=class_names
    )

    return model, train_accuracy, test_accuracy, conf_matrix, confusion_df


def classify_sample(model, raw_values):
    """raw_values = [CulmenLength, CulmenDepth, FlipperLength, BodyMass,
    OriginLocation]. Returns (class name, output vector)."""

    location = str(raw_values[4]).strip().title()

    if location not in location_map:
        raise ValueError(f"Unknown location. Valid: {list(location_map)}")

    x = np.array(
        [float(v) for v in raw_values[:4]] + [location_map[location]],
        dtype=float
    )

    # use the TRAINING mean and std
    x = (x - mean.to_numpy()) / std.to_numpy()

    output = model.forward(x)

    return class_names[int(np.argmax(output))], output.flatten()


def tune_hyperparameters():
    """Grid search. Prints the best settings for each activation function.

    Each network is trained once up to the largest epoch value and evaluated
    at the checkpoints, so several epoch values are tested without retraining.
    Run with:  python code.py tune
    """

    architectures = [[5], [10], [5, 5], [10, 5]]
    learning_rates = [0.001, 0.01, 0.05, 0.1]
    checkpoints = [100, 300, 500, 1000]      # epoch values to evaluate

    combos = list(itertools.product(
        ["sigmoid", "tanh"], [True, False], architectures, learning_rates))

    results = []
    start = time.time()

    for n, (act, bias, hidden, lr) in enumerate(combos, start=1):

        model = MLP(X_train.shape[1], hidden, len(class_names),
                    lr, act, bias)

        done = 0
        for cp in checkpoints:
            # continue training from where the previous checkpoint stopped
            with contextlib.redirect_stdout(io.StringIO()):
                model.train(X_train, y_train, cp - done)
            done = cp

            train_acc = np.mean(model.predict(X_train) == y_train) * 100
            test_acc = np.mean(model.predict(X_test) == y_test) * 100

            results.append({
                "Activation": act,
                "TrainAcc": round(train_acc, 2),
                "TestAcc": round(test_acc, 2),
                "LR": lr,
                "Epochs": cp,
                "Layers": len(hidden),
                "HiddenNodes": ",".join(map(str, hidden)),
                "Bias": "Yes" if bias else "No",
            })

        print(f"[{n}/{len(combos)}] {act} bias={bias} {hidden} lr={lr} "
              f"({time.time() - start:.0f}s elapsed)")

    df = pd.DataFrame(results)
    df.to_csv("tuning_results.csv", index=False)

    for act in ["sigmoid", "tanh"]:
        sub = df[df["Activation"] == act].sort_values(
            ["TestAcc", "TrainAcc", "Epochs"],
            ascending=[False, False, True])

        print(f"\n===== TOP 10 for {act.upper()} =====")
        print(sub.head(10).to_string(index=False))


if __name__ == "__main__":

    # python code.py tune  ->  run the hyperparameter search instead
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        tune_hyperparameters()
        sys.exit()

    num_hidden_layers = int(input("Enter number of hidden layers: "))

    hidden_layers = []

    for i in range(num_hidden_layers):

        neurons = int(input(f"Enter number of neurons in hidden layer {i + 1}: "))
        hidden_layers.append(neurons)

    learning_rate = float(input("Enter learning rate (eta): "))
    epochs = int(input("Enter number of epochs: "))
    bias_choice = input("Use bias? (yes/no): ").lower()

    while bias_choice not in ["yes", "no"]:
        bias_choice = input("Please enter yes or no: ").lower()

    use_bias = bias_choice == "yes"

    activation_name = input("Choose activation function (sigmoid/tanh): ").lower()

    while activation_name not in ["sigmoid", "tanh"]:
        activation_name = input("Please enter sigmoid or tanh: ").lower()

    model, train_accuracy, test_accuracy, conf_matrix, confusion_df = run_model(
        hidden_layers, learning_rate, epochs, activation_name, use_bias
    )

    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Testing Accuracy: {test_accuracy:.2f}%")
    print("Confusion Matrix:")
    print(confusion_df)
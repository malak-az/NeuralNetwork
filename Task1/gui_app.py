import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from itertools import combinations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


from models import NN, confusion_matrix, accuracy
from el_sho8l import FEATURES, penguins, load_preprocess, split, standardize



class PenguinApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perceptron and Adaline - Penguins")
        self.geometry("1100x720")

        self.model = None
        self.trained = False
        self.tested = False
        self.new_point = None
        self.load_dataset(penguins)
        self.make_interface()

    def load_dataset(self, path):
        # loading / preprocessing lives in el_sho8l.py (no UI code there)
        data = load_preprocess(path)
        classes = sorted(data["Species"].dropna().unique())
        if len(classes) < 2:
            raise ValueError("The dataset must contain at least two species.")
        self.data, self.classes, self.path = data, classes, path
        self.class_pairs = [f"{a} / {b}" for a, b in combinations(classes, 2)]

    def make_interface(self):
        left = ttk.Frame(self, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(self, padding=10)
        right.pack(side="right", fill="both", expand=True)

        self.feature1 = tk.StringVar(value=FEATURES[0])
        self.feature2 = tk.StringVar(value=FEATURES[2])
        self.class_pair = tk.StringVar(value=self.class_pairs[0])
        self.lr = tk.StringVar(value="0.01")
        self.epochs = tk.StringVar(value="100")
        self.threshold = tk.StringVar(value="0.1")
        self.use_bias = tk.BooleanVar(value=True)
        self.algorithm = tk.StringVar(value="perceptron")
        self.sample1 = tk.StringVar()
        self.sample2 = tk.StringVar()

        settings = ttk.LabelFrame(left, text="Settings", padding=8)
        settings.pack(fill="x")

        options = [
            ("Feature 1", self.feature1, FEATURES),
            ("Feature 2", self.feature2, FEATURES),
            ("Species pair", self.class_pair, self.class_pairs),
        ]
        for row, (label, variable, values) in enumerate(options):
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky="w", pady=3)
            box = ttk.Combobox(settings, textvariable=variable, values=values,
                               state="readonly", width=20)
            box.grid(row=row, column=1, sticky="ew", pady=3)

        for row, (label, variable) in enumerate([
            ("Learning rate", self.lr),
            ("Epochs", self.epochs),
            ("MSE threshold", self.threshold),
        ], start=3):
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(settings, textvariable=variable, width=12).grid(row=row, column=1, sticky="ew", pady=3)

        ttk.Checkbutton(settings, text="Use bias", variable=self.use_bias).grid(
            row=6, column=0, columnspan=2, sticky="w"
        )
        ttk.Radiobutton(settings, text="Perceptron", variable=self.algorithm,
                        value="perceptron").grid(row=7, column=0, sticky="w")
        ttk.Radiobutton(settings, text="Adaline", variable=self.algorithm,
                        value="adaline").grid(row=7, column=1, sticky="w")

        ttk.Button(left, text="Train", command=self.train).pack(fill="x", pady=(8, 3))
        ttk.Button(left, text="Test", command=self.test).pack(fill="x", pady=3)

        self.weights_label = ttk.Label(left, text="Weights and bias will appear here",
                                       wraplength=260)
        self.weights_label.pack(anchor="w", pady=8)

        sample_box = ttk.LabelFrame(left, text="Classify a new sample", padding=8)
        sample_box.pack(fill="x", pady=5)
        self.sample_label1 = ttk.Label(sample_box, text=self.feature1.get())
        self.sample_label1.grid(row=0, column=0, sticky="w")
        self.sample_label2 = ttk.Label(sample_box, text=self.feature2.get())
        self.sample_label2.grid(row=1, column=0, sticky="w")
        ttk.Entry(sample_box, textvariable=self.sample1, width=12).grid(row=0, column=1)
        ttk.Entry(sample_box, textvariable=self.sample2, width=12).grid(row=1, column=1)
        ttk.Button(sample_box, text="Classify", command=self.classify).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=5
        )
        self.class_result = ttk.Label(sample_box, text="Train a model first.")
        self.class_result.grid(row=3, column=0, columnspan=2, sticky="w")

        self.result_label = ttk.Label(right, text="Train the model, then press Test.",
                                      font=("Courier", 10), justify="left")
        self.result_label.pack(anchor="w")

        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.figure.add_subplot(211)
        self.mse_ax = self.figure.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.draw_empty()

    def train(self):
        try:
            f1, f2 = self.feature1.get(), self.feature2.get()
            if f1 == f2:
                raise ValueError("Choose two different features.")

            lr = float(self.lr.get())
            epochs = int(self.epochs.get())
            threshold = float(self.threshold.get())
            if lr <= 0 or epochs < 1 or threshold < 0:
                raise ValueError("Use a positive learning rate, at least 1 epoch, and a non-negative threshold.")

        except ValueError as error:
            messagebox.showwarning("Invalid settings", str(error))
            return

        class_a, class_b = self.class_pair.get().split(" / ")

        # 30 train / 20 test per class and the scaling come from el_sho8l.py
        try:
            self.X_train_raw, self.y_train, self.X_test_raw, self.y_test = split(
                self.data, class_a, class_b, f1, f2)
        except ValueError as error:
            messagebox.showwarning("Cannot split data", str(error))
            return
        self.X_train, self.X_test, self.mean, self.scale = standardize(
            self.X_train_raw, self.X_test_raw)
        self.class_a, self.class_b = class_a, class_b
        self.feature_names = (f1, f2)

        self.model = NN(lr=lr, epochs=epochs, bias=self.use_bias.get(), threshold=threshold)
        # remember which algorithm trained this model (the radio button may change later)
        self.trained_algorithm = self.algorithm.get()
        if self.trained_algorithm == "perceptron":
            self.model.perc(self.X_train, self.y_train)
        else:
            self.model.adaline(self.X_train, self.y_train)

        self.trained = True
        self.tested = False
        self.new_point = None
        self.sample_label1.config(text=f1)
        self.sample_label2.config(text=f2)
        self.weights_label.config(
            text=f"Weights: w1={self.model.w[0]:+.4f}, w2={self.model.w[1]:+.4f}\n"
                 f"Bias: {self.model.b:+.4f}"
        )
        self.class_result.config(text="Enter feature values and press Classify.")
        self.result_label.config(text="Model trained. Press Test to evaluate.")
        self.draw_results()

    def test(self):
        if not self.trained:
            messagebox.showinfo("Train first", "Train the model before testing.")
            return

        predictions = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, predictions)
        score = accuracy(cm)
        self.tested = True
        a, b = self.class_a, self.class_b
        self.result_label.config(
            text=f"Confusion matrix (rows = actual, columns = predicted)\n"
                 f"{'':>12}{a:>12}{b:>12}\n"
                 f"{a:>12}{cm[0, 0]:>12}{cm[0, 1]:>12}\n"
                 f"{b:>12}{cm[1, 0]:>12}{cm[1, 1]:>12}\n"
                 f"Overall accuracy: {score:.1%}"
        )
        self.draw_results()

    def classify(self):
        if not self.trained:
            messagebox.showinfo("Train first", "Train the model before classifying.")
            return
        try:
            point = np.array([float(self.sample1.get()), float(self.sample2.get())])
        except ValueError:
            messagebox.showwarning("Invalid sample", "Enter a number for each feature.")
            return

        scaled = (point - self.mean) / self.scale
        prediction = self.model.predict(scaled.reshape(1, -1))[0]
        name = self.class_a if prediction == 1 else self.class_b
        self.class_result.config(text=f"Predicted species: {name}")
        self.new_point = point
        self.draw_results()

    def open_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            self.load_dataset(Path(path))
        except Exception as error:
            messagebox.showerror("Could not load CSV", str(error))
            return
        self.trained = self.tested = False
        self.new_point = None
        self.pair_box["values"] = self.class_pairs
        self.class_pair.set(self.class_pairs[0])
        self.weights_label.config(text="Weights and bias will appear here")
        self.class_result.config(text="Train a model first.")
        self.result_label.config(text="Train the model, then press Test.")
        self.draw_empty()
        messagebox.showinfo("Dataset loaded", f"Loaded {len(self.data)} rows.")

    def draw_empty(self):
        self.ax.clear()
        self.mse_ax.clear()
        self.ax.set_title("Decision boundary")
        self.mse_ax.set_title("Adaline MSE")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def draw_results(self):
        if not self.trained:
            self.draw_empty()
            return

        self.ax.clear()
        for label, name in [(1, self.class_a), (-1, self.class_b)]:
            points = self.X_train_raw[self.y_train == label]
            self.ax.scatter(points[:, 0], points[:, 1], label=f"{name} train")

            if self.tested:
                points = self.X_test_raw[self.y_test == label]
                self.ax.scatter(points[:, 0], points[:, 1], marker="x", label=f"{name} test")

        # Convert the decision boundary back to the original feature scale.
        w = self.model.w / self.scale
        b = self.model.b - np.dot(self.model.w, self.mean / self.scale)
        points = [self.X_train_raw, self.X_test_raw]
        if self.new_point is not None:
            points.append(self.new_point.reshape(1, -1))
        (x_lo, y_lo), (x_hi, y_hi) = np.vstack(points).min(axis=0), np.vstack(points).max(axis=0)
        x_pad, y_pad = max(0.08 * (x_hi - x_lo), 0.5), max(0.08 * (y_hi - y_lo), 0.5)
        xlim, ylim = (x_lo - x_pad, x_hi + x_pad), (y_lo - y_pad, y_hi + y_pad)

        if abs(w[1]) > 1e-12:
            xs = np.array(xlim)
            self.ax.plot(xs, -(w[0] * xs + b) / w[1], "k-", label="Boundary")
        elif abs(w[0]) > 1e-12:
            self.ax.axvline(-b / w[0], color="black", label="Boundary")

        if self.new_point is not None:
            self.ax.scatter(self.new_point[0], self.new_point[1], marker="*", s=180,
                            color="gold", edgecolors="black", label="New sample")

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel(self.feature_names[0])
        self.ax.set_ylabel(self.feature_names[1])
        self.ax.set_title(f"{self.trained_algorithm.title()}: {self.class_a} vs {self.class_b}")
        self.ax.legend(fontsize=8)
        self.ax.grid(alpha=0.25)

        self.mse_ax.clear()
        if self.trained_algorithm == "adaline":
            self.mse_ax.set_title("Adaline MSE per epoch")
            self.mse_ax.plot(self.model.all_mse)
            self.mse_ax.set_xlabel("Epoch")
            self.mse_ax.set_ylabel("MSE")
        else:
            self.mse_ax.set_title("MSE")
            self.mse_ax.text(0.5, 0.5, "MSE is only used by Adaline",
                             ha="center", va="center",
                             transform=self.mse_ax.transAxes)
        self.figure.tight_layout()
        self.canvas.draw_idle()


if __name__ == "__main__":
    try:
        app = PenguinApp()
        app.mainloop()
    except Exception as error:
        messagebox.showerror("Application error", str(error))

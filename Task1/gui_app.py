"""Tkinter GUI for the existing Task1 Perceptron/Adaline implementation.

The algorithms in models.py are intentionally left unchanged. This file is
only the interface and the small amount of glue needed to call them.
Run from the repository root with: python Task1/gui_app.py
"""
from __future__ import annotations

import itertools
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:  # supports both `python Task1/gui_app.py` and package imports
    from models import perceptron
except ImportError:
    from .models import perceptron

FEATURES = ["CulmenLength", "CulmenDepth", "FlipperLength", "OriginLocation", "BodyMass"]
CATEGORICAL_FEATURE = "OriginLocation"
CLASS_COLOURS = ("#1f77b4", "#ff7f0e")
DEFAULT_CSV = Path(__file__).resolve().parent / "data" / "penguins.csv"


class PenguinApp(tk.Tk):
    def __init__(self, csv_path: Path = DEFAULT_CSV):
        super().__init__()
        self.title("Neural Network Lab - Perceptron & Adaline (Penguins)")
        self.geometry("1220x780")
        self.minsize(1000, 680)
        self.csv_path = Path(csv_path)
        self.model = None
        self._tested = False
        self._classified = None
        self._trained = False
        self._load_data(self.csv_path)
        self._build_variables()
        self._build_menu()
        self._build_layout()
        self._refresh_dataset_widgets()
        self._draw_empty()

    def _load_data(self, path: Path):
        frame = pd.read_csv(path, na_values=["NA", "", " "])
        required = ["Species", *FEATURES]
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
        self.frame = frame[required].copy()
        self.origin_codes = {name: i for i, name in enumerate(sorted(self.frame[CATEGORICAL_FEATURE].dropna().unique()))}
        self.frame[CATEGORICAL_FEATURE] = self.frame[CATEGORICAL_FEATURE].map(self.origin_codes)
        for col in FEATURES:
            self.frame[col] = pd.to_numeric(self.frame[col], errors="coerce")
        self.frame = self.frame.dropna(subset=required).reset_index(drop=True)
        self.class_names = sorted(self.frame["Species"].unique().tolist())
        if len(self.class_names) < 2:
            raise ValueError("The dataset must contain at least two species.")
        self.csv_path = path

    def _build_variables(self):
        self.feature1 = tk.StringVar(value="CulmenLength")
        self.feature2 = tk.StringVar(value="FlipperLength")
        self.class_pair = tk.StringVar()
        self.eta = tk.StringVar(value="0.01")
        self.epochs = tk.StringVar(value="100")
        self.normalize = tk.BooleanVar(value=True)
        self.algorithm = tk.StringVar(value="perceptron")
        self.sample_x = tk.StringVar()
        self.sample_y = tk.StringVar()
        self.status = tk.StringVar(value="Choose settings and press Train.")

    def _build_menu(self):
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open dataset (CSV)...", command=self._open_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)

    def _build_layout(self):
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W,
                  padding=(8, 2)).pack(side=tk.BOTTOM, fill=tk.X)
        left = ttk.Frame(self, padding=8)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self, padding=(0, 8, 8, 8))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_input_frame(left)
        self._build_init_frame(left)
        self._build_buttons(left)
        self._build_classification_frame(left)
        self._build_results(right)
        self._build_plot(right)

    def _build_input_frame(self, parent):
        box = ttk.LabelFrame(parent, text="1. User Input", padding=8)
        box.pack(fill=tk.X)
        box.columnconfigure(1, weight=1)
        def row(r, label, widget):
            ttk.Label(box, text=label).grid(row=r, column=0, sticky=tk.W, pady=2)
            widget.grid(row=r, column=1, sticky=tk.EW, padx=(8, 0), pady=2)
        row(0, "Feature 1", ttk.Combobox(box, textvariable=self.feature1, values=FEATURES, state="readonly", width=18))
        row(1, "Feature 2", ttk.Combobox(box, textvariable=self.feature2, values=FEATURES, state="readonly", width=18))
        self.class_box = ttk.Combobox(box, textvariable=self.class_pair, state="readonly", width=25)
        row(2, "Classes", self.class_box)
        row(3, "Learning rate (eta)", ttk.Entry(box, textvariable=self.eta, width=10))
        row(4, "Epochs (m)", ttk.Entry(box, textvariable=self.epochs, width=10))
        ttk.Checkbutton(box, text="Normalize features (recommended)", variable=self.normalize).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        algo = ttk.Frame(box)
        algo.grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        ttk.Label(algo, text="Algorithm:").pack(side=tk.LEFT)
        ttk.Radiobutton(algo, text="Perceptron", value="perceptron", variable=self.algorithm).pack(side=tk.LEFT, padx=(8, 4))
        ttk.Radiobutton(algo, text="Adaline", value="adaline", variable=self.algorithm).pack(side=tk.LEFT)

    def _build_init_frame(self, parent):
        box = ttk.LabelFrame(parent, text="2. Initialization", padding=8)
        box.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(box, text="Number of features = 2\nNumber of classes = 2\nBias is included by models.py").pack(anchor=tk.W)
        self.weights_label = ttk.Label(box, text="Weights and bias appear after training", justify=tk.LEFT)
        self.weights_label.pack(anchor=tk.W, pady=(4, 0))

    def _build_buttons(self, parent):
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X, pady=8)
        ttk.Button(bar, text="Train", command=self._on_train).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        self.test_btn = ttk.Button(bar, text="Test", command=self._on_test, state=tk.DISABLED)
        self.test_btn.pack(side=tk.LEFT, expand=True, fill=tk.X)

    def _build_classification_frame(self, parent):
        box = ttk.LabelFrame(parent, text="3. Classification", padding=8)
        box.pack(fill=tk.X)
        box.columnconfigure(1, weight=1)
        self.sample_x_label = ttk.Label(box, text="Feature 1")
        self.sample_y_label = ttk.Label(box, text="Feature 2")
        self.sample_x_label.grid(row=0, column=0, sticky=tk.W, pady=2)
        self.sample_y_label.grid(row=1, column=0, sticky=tk.W, pady=2)
        self.sample_x_entry = ttk.Entry(box, textvariable=self.sample_x, state=tk.DISABLED)
        self.sample_y_entry = ttk.Entry(box, textvariable=self.sample_y, state=tk.DISABLED)
        self.sample_x_entry.grid(row=0, column=1, sticky=tk.EW, padx=(8, 0), pady=2)
        self.sample_y_entry.grid(row=1, column=1, sticky=tk.EW, padx=(8, 0), pady=2)
        self.classify_btn = ttk.Button(box, text="Classify sample", state=tk.DISABLED, command=self._on_classify)
        self.classify_btn.grid(row=2, column=0, columnspan=2, sticky=tk.EW, pady=(6, 2))
        self.classify_result = ttk.Label(box, text="Train a model first.", font=("TkDefaultFont", 10, "bold"))
        self.classify_result.grid(row=3, column=0, columnspan=2, sticky=tk.W)

    def _build_plot(self, parent):
        self.figure = Figure(figsize=(7, 5.4), dpi=100)
        grid = self.figure.add_gridspec(2, 1, height_ratios=(3, 1.2), hspace=0.5, left=0.17, right=0.97, top=0.94, bottom=0.09)
        self.ax_main = self.figure.add_subplot(grid[0])
        self.ax_mse = self.figure.add_subplot(grid[1])
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_results(self, parent):
        box = ttk.LabelFrame(parent, text="Test results (remaining samples)", padding=8)
        box.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))
        self.cm_frame = ttk.Frame(box)
        self.cm_frame.pack(side=tk.LEFT)
        self.accuracy_label = ttk.Label(box, text="Overall accuracy: -", font=("TkDefaultFont", 14, "bold"))
        self.accuracy_label.pack(side=tk.LEFT, padx=40)
        self._show_confusion(None, None)

    def _refresh_dataset_widgets(self):
        self._pair_indices = list(itertools.combinations(range(len(self.class_names)), 2))
        labels = [f"{self.class_names[a]} / {self.class_names[b]}" for a, b in self._pair_indices]
        self.class_box["values"] = labels
        self.class_box.current(0)

    def _open_dataset(self):
        path = filedialog.askopenfilename(title="Open penguins CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            self._load_data(Path(path))
        except Exception as exc:
            messagebox.showerror("Could not load dataset", str(exc))
            return
        self._refresh_dataset_widgets()
        self._reset_after_new_state()
        self.status.set(f"Loaded {len(self.frame)} usable rows from {Path(path).name}.")

    def _selected_classes(self):
        pos = self.class_box.current()
        if pos < 0:
            raise ValueError("Choose two classes.")
        a, b = self._pair_indices[pos]
        return self.class_names[a], self.class_names[b]

    def _feature_values(self, frame, feature):
        values = frame[feature].to_numpy(dtype=float)
        if feature == CATEGORICAL_FEATURE:
            # Origin codes are learned from the loaded dataset and used consistently.
            return values
        return values

    def _on_train(self):
        try:
            f1, f2 = self.feature1.get(), self.feature2.get()
            if f1 == f2:
                raise ValueError("Choose two different features.")
            eta, epochs = float(self.eta.get()), int(self.epochs.get())
            if eta <= 0 or epochs < 1:
                raise ValueError("Learning rate must be positive and epochs must be at least 1.")
            class_a, class_b = self._selected_classes()
        except (ValueError, IndexError) as exc:
            messagebox.showwarning("Invalid input", str(exc))
            return
        selected = self.frame[self.frame["Species"].isin([class_a, class_b])].copy()
        X = selected[[f1, f2]].to_numpy(dtype=float)
        # Match the existing models.py convention: +1 for class A and -1 for class B.
        y = np.where(selected["Species"].to_numpy() == class_a, 1.0, -1.0)
        rng = np.random.default_rng(42)
        train_idx, test_idx = [], []
        for label in (1.0, -1.0):
            idx = np.flatnonzero(y == label)
            rng.shuffle(idx)
            n_train = max(1, int(round(len(idx) * 0.7)))
            train_idx.extend(idx[:n_train])
            test_idx.extend(idx[n_train:])
        train_idx, test_idx = np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)
        rng.shuffle(train_idx); rng.shuffle(test_idx)
        self.mean = X[train_idx].mean(axis=0) if self.normalize.get() else np.zeros(2)
        self.scale = X[train_idx].std(axis=0) if self.normalize.get() else np.ones(2)
        self.scale[self.scale == 0] = 1.0
        self.X_train_raw, self.X_test_raw = X[train_idx], X[test_idx]
        self.X_train = (self.X_train_raw - self.mean) / self.scale
        self.X_test = (self.X_test_raw - self.mean) / self.scale
        self.y_train, self.y_test = y[train_idx], y[test_idx]
        self.train_species, self.test_species = selected["Species"].to_numpy()[train_idx], selected["Species"].to_numpy()[test_idx]
        self.class_a, self.class_b = class_a, class_b
        self.feature_names = (f1, f2)
        self.model = perceptron(learning_rate=eta, mx_epochs=epochs)
        initial_weights = None
        if self.algorithm.get() == "perceptron":
            self.model.perc(self.X_train, self.y_train)
        else:
            self.model.adaline(self.X_train, self.y_train)
        self._tested = False
        self._classified = None
        self._trained = True
        self.test_btn.config(state=tk.NORMAL)
        self.classify_btn.config(state=tk.NORMAL)
        self.sample_x_entry.config(state=tk.NORMAL); self.sample_y_entry.config(state=tk.NORMAL)
        self.sample_x_label.config(text=f1); self.sample_y_label.config(text=f2)
        self.sample_x.set(""); self.sample_y.set("")
        self.classify_result.config(text="Enter a sample and press Classify.")
        self._show_confusion(None, None)
        self._draw_results()
        self.weights_label.config(text=f"Final weights: [{self.model.w[0]:+.4f}, {self.model.w[1]:+.4f}]\nBias: {self.model.b:+.4f}")
        train_pred = self._predict(self.X_train)
        acc = float(np.mean(train_pred == self.y_train))
        self.status.set(f"{self.algorithm.get().title()} trained on {len(train_idx)} samples; training accuracy = {acc:.1%}. Press Test to evaluate.")

    def _predict(self, X):
        if self.algorithm.get() == "perceptron":
            return self.model.perc_predict(X)
        return np.where(self.model.adaline_predict(X) >= 0, 1.0, -1.0)

    def _on_test(self):
        if not self._trained:
            return
        pred = self._predict(self.X_test)
        matrix = np.zeros((2, 2), dtype=int)
        for actual, guess in zip(self.y_test, pred):
            matrix[0 if actual == 1 else 1, 0 if guess == 1 else 1] += 1
        accuracy = float(np.mean(pred == self.y_test)) if len(pred) else 0.0
        self._tested = True
        self._show_confusion(matrix, [self.class_a, self.class_b])
        self.accuracy_label.config(text=f"Overall accuracy: {accuracy:.1%}")
        self._draw_results()
        self.status.set(f"Tested on {len(self.y_test)} unseen samples - accuracy {accuracy:.1%}.")

    def _on_classify(self):
        try:
            sample = np.array([float(self.sample_x.get()), float(self.sample_y.get())])
        except ValueError:
            messagebox.showwarning("Invalid sample", "Enter numeric values for both selected features.")
            return
        scaled = (sample - self.mean) / self.scale
        result = self._predict(scaled.reshape(1, -1))[0]
        name = self.class_a if result == 1 else self.class_b
        class_idx = self.class_names.index(name)
        self.classify_result.config(text=f"Predicted species: {name}")
        self._classified = (sample[0], sample[1], class_idx)
        self._draw_results()

    def _reset_after_new_state(self):
        self._trained = False
        self.test_btn.config(state=tk.DISABLED)
        self.classify_btn.config(state=tk.DISABLED)
        self.sample_x_entry.config(state=tk.DISABLED); self.sample_y_entry.config(state=tk.DISABLED)
        self.classify_result.config(text="Train a model first.")
        self._show_confusion(None, None)
        self._draw_empty()

    def _show_confusion(self, matrix, names):
        for child in self.cm_frame.winfo_children():
            child.destroy()
        if matrix is None:
            self.accuracy_label.config(text="Overall accuracy: -")
            ttk.Label(self.cm_frame, text="Press Test to build the confusion matrix.").pack()
            return
        for c, label in enumerate(("Actual \\ Predicted", *names)):
            ttk.Label(self.cm_frame, text=label, font=("TkDefaultFont", 9, "bold"), padding=(8, 3)).grid(row=0, column=c)
        for r, name in enumerate(names, start=1):
            ttk.Label(self.cm_frame, text=name, font=("TkDefaultFont", 9, "bold"), padding=(8, 3)).grid(row=r, column=0, sticky=tk.W)
            for c in range(2):
                tk.Label(self.cm_frame, text=str(matrix[r - 1, c]), width=8, pady=4, relief=tk.GROOVE,
                         bg="#d8f0d8" if r - 1 == c else "#f6d8d8").grid(row=r, column=c + 1, padx=1, pady=1)

    def _draw_empty(self):
        self.ax_main.clear(); self.ax_mse.clear()
        self.ax_main.set_title("Decision boundary - press Train")
        self.ax_mse.set_title("Training history")
        self.ax_mse.text(0.5, 0.5, "The existing models.py does not record per-epoch MSE.", ha="center", va="center", transform=self.ax_mse.transAxes, fontsize=8)
        self.ax_mse.set_xticks([]); self.ax_mse.set_yticks([])
        self.canvas.draw_idle()

    def _set_axis_labels(self, axis, feature, which):
        (axis.set_xlabel if which == "x" else axis.set_ylabel)(feature)
        if feature == CATEGORICAL_FEATURE and self.origin_codes:
            codes = list(self.origin_codes.values()); labels = list(self.origin_codes.keys())
            (axis.set_xticks if which == "x" else axis.set_yticks)(codes)
            (axis.set_xticklabels if which == "x" else axis.set_yticklabels)(labels)

    def _draw_results(self):
        if not self._trained:
            self._draw_empty(); return
        ax = self.ax_main
        ax.clear()
        for label, colour, name in ((1.0, CLASS_COLOURS[0], self.class_a), (-1.0, CLASS_COLOURS[1], self.class_b)):
            pts = self.X_train_raw[self.y_train == label]
            ax.scatter(pts[:, 0], pts[:, 1], c=colour, s=28, label=f"{name} (train)")
            if self._tested:
                pts = self.X_test_raw[self.y_test == label]
                ax.scatter(pts[:, 0], pts[:, 1], c=colour, marker="x", s=40, label=f"{name} (test)")
        all_pts = np.vstack([self.X_train_raw, self.X_test_raw])
        (x_lo, y_lo), (x_hi, y_hi) = all_pts.min(axis=0), all_pts.max(axis=0)
        pad_x, pad_y = max(0.08 * (x_hi - x_lo), 0.5), max(0.08 * (y_hi - y_lo), 0.5)
        xlim, ylim = (x_lo - pad_x, x_hi + pad_x), (y_lo - pad_y, y_hi + pad_y)
        # Convert the learned line from normalized to original feature units.
        w = self.model.w / self.scale
        b = self.model.b - float(np.dot(self.model.w, self.mean / self.scale))
        if abs(w[1]) > 1e-12:
            xs = np.array(xlim)
            ax.plot(xs, -(w[0] * xs + b) / w[1], color="black", linewidth=1.6, label="Decision boundary")
        elif abs(w[0]) > 1e-12:
            ax.axvline(-b / w[0], color="black", linewidth=1.6, label="Decision boundary")
        if self._classified is not None:
            sx, sy, class_idx = self._classified
            ax.scatter([sx], [sy], marker="*", s=260, c="gold", edgecolors="black", zorder=5, label=f"New sample -> {self.class_names[class_idx]}")
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        self._set_axis_labels(ax, self.feature_names[0], "x"); self._set_axis_labels(ax, self.feature_names[1], "y")
        ax.set_title(f"{self.algorithm.get().title()}: {self.class_a} vs {self.class_b}")
        ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.25)
        self.ax_mse.clear()
        self.ax_mse.set_title("Training history")
        self.ax_mse.text(0.5, 0.5, "Per-epoch MSE is not exposed by the existing models.py.\nYour algorithm code has been left unchanged.", ha="center", va="center", transform=self.ax_mse.transAxes, fontsize=9)
        self.ax_mse.set_xticks([]); self.ax_mse.set_yticks([])
        self.canvas.draw_idle()


def main():
    try:
        app = PenguinApp(DEFAULT_CSV)
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Could not load dataset", str(exc), parent=root)
        root.destroy()
        return
    app.mainloop()


if __name__ == "__main__":
    main()

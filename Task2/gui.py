"""
gui.py - Tkinter GUI for the Back-Propagation MLP (penguins data only)

Run:  python gui.py
Keep this file in the same folder as code.py.
Requires matplotlib:  pip install matplotlib
"""

import textwrap
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except ImportError:
    _root = tk.Tk()
    _root.withdraw()
    messagebox.showerror("matplotlib missing",
                         "Please install matplotlib:\n\npip install matplotlib")
    raise

try:
    import code as nn   # code.py (loads and preprocesses the penguins data)
except Exception as e:
    _root = tk.Tk()
    _root.withdraw()
    messagebox.showerror("Could not load code.py",
                         f"{type(e).__name__}: {e}\n\n"
                         "Check the penguins.csv path inside code.py.")
    raise


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Back-Propagation Neural Network - Penguins")
        self.geometry("1200x700")
        self.minsize(1050, 640)

        self.model = None
        self.run_count = 0
        self.neuron_vars = []

        self._build_settings_panel()
        self._build_right_panel()
        self._rebuild_neuron_entries()
        self._draw_placeholder()

    # ==================================================================
    # Left panel: user input
    # ==================================================================
    def _build_settings_panel(self):
        left = ttk.LabelFrame(self, text="Network settings", padding=10)
        left.pack(side="left", fill="y", padx=10, pady=10)

        # Number of hidden layers
        ttk.Label(left, text="Number of hidden layers:").grid(
            row=0, column=0, sticky="w")
        self.layers_var = tk.StringVar(value="1")
        ttk.Spinbox(left, from_=1, to=10, width=6,
                    textvariable=self.layers_var,
                    command=self._rebuild_neuron_entries).grid(
            row=0, column=1, pady=2)
        self.layers_var.trace_add("write", self._rebuild_neuron_entries)

        # Neurons in each hidden layer (built dynamically)
        self.neuron_frame = ttk.LabelFrame(
            left, text="Neurons in each hidden layer", padding=6)
        self.neuron_frame.grid(row=1, column=0, columnspan=2,
                               sticky="we", pady=8)

        # Learning rate, epochs
        ttk.Label(left, text="Learning rate (eta):").grid(
            row=2, column=0, sticky="w")
        self.eta_var = tk.StringVar(value="0.01")
        ttk.Entry(left, textvariable=self.eta_var, width=8).grid(
            row=2, column=1, pady=2)

        ttk.Label(left, text="Number of epochs (m):").grid(
            row=3, column=0, sticky="w")
        self.epochs_var = tk.StringVar(value="500")
        ttk.Entry(left, textvariable=self.epochs_var, width=8).grid(
            row=3, column=1, pady=2)

        # Bias checkbox
        self.bias_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Add bias", variable=self.bias_var).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=6)

        # Activation function
        act = ttk.LabelFrame(left, text="Activation function", padding=6)
        act.grid(row=5, column=0, columnspan=2, sticky="we", pady=4)
        self.act_var = tk.StringVar(value="sigmoid")
        ttk.Radiobutton(act, text="Sigmoid", value="sigmoid",
                        variable=self.act_var).pack(anchor="w")
        ttk.Radiobutton(act, text="Hyperbolic Tangent (tanh)", value="tanh",
                        variable=self.act_var).pack(anchor="w")

        # Buttons + status
        self.train_btn = ttk.Button(left, text="Train && Test",
                                    command=self._train)
        self.train_btn.grid(row=6, column=0, columnspan=2,
                            sticky="we", pady=(12, 4))

        ttk.Button(left, text="Save results image...",
                   command=self._save_figure).grid(
            row=7, column=0, columnspan=2, sticky="we", pady=2)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(left, textvariable=self.status_var, wraplength=230,
                  foreground="#444").grid(row=8, column=0, columnspan=2,
                                          sticky="w", pady=(8, 0))

    def _rebuild_neuron_entries(self, *_):
        try:
            n = int(self.layers_var.get())
        except ValueError:
            return
        n = max(1, min(10, n))

        old = [v.get() for v in self.neuron_vars]
        for w in self.neuron_frame.winfo_children():
            w.destroy()

        self.neuron_vars = []
        for i in range(n):
            var = tk.StringVar(value=old[i] if i < len(old) else "5")
            ttk.Label(self.neuron_frame, text=f"Layer {i + 1}:").grid(
                row=i, column=0, sticky="w", padx=(0, 8), pady=1)
            ttk.Entry(self.neuron_frame, textvariable=var, width=8).grid(
                row=i, column=1, pady=1)
            self.neuron_vars.append(var)

    # ==================================================================
    # Right panel: visual results / classification
    # ==================================================================
    def _build_right_panel(self):
        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True,
                   padx=(0, 10), pady=10)

        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)

        # --- Results tab: summary line + figure ---
        res_tab = ttk.Frame(self.tabs)
        self.tabs.add(res_tab, text="Results")

        self.summary_var = tk.StringVar(value="")
        ttk.Label(res_tab, textvariable=self.summary_var,
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8,
                                                      pady=(8, 0))

        self.figure = Figure(figsize=(8, 5), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=res_tab)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- Classification tab ---
        cls_tab = ttk.Frame(self.tabs, padding=15)
        self.tabs.add(cls_tab, text="Classify a sample")

        ttk.Label(cls_tab, text="Enter the 5 features of one sample:",
                  font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        self.sample_vars = {}
        numeric_names = ["CulmenLength", "CulmenDepth",
                         "FlipperLength", "BodyMass"]
        for r, name in enumerate(numeric_names, start=1):
            ttk.Label(cls_tab, text=name + ":").grid(
                row=r, column=0, sticky="w", pady=3)
            var = tk.StringVar()
            ttk.Entry(cls_tab, textvariable=var, width=16).grid(
                row=r, column=1, sticky="w", pady=3, padx=8)
            self.sample_vars[name] = var

        ttk.Label(cls_tab, text="OriginLocation:").grid(
            row=5, column=0, sticky="w", pady=3)
        self.location_var = tk.StringVar()
        ttk.Combobox(cls_tab, textvariable=self.location_var,
                     state="readonly", width=14,
                     values=list(nn.location_map)).grid(
            row=5, column=1, sticky="w", pady=3, padx=8)

        ttk.Button(cls_tab, text="Classify", command=self._classify).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=14)

        self.class_result_var = tk.StringVar(value="Train the network first.")
        ttk.Label(cls_tab, textvariable=self.class_result_var,
                  font=("Segoe UI", 11), justify="left").grid(
            row=7, column=0, columnspan=2, sticky="w")

    # ==================================================================
    # Drawing
    # ==================================================================
    def _draw_placeholder(self):
        self.figure.clear()
        self.figure.text(0.5, 0.5, "Train the network to see the results",
                         ha="center", va="center", fontsize=14, color="gray")
        self.canvas.draw()

    def _draw_results(self, conf_matrix, history, overall, title_extra):
        self.figure.clear()
        gs = self.figure.add_gridspec(2, 2, width_ratios=[1.15, 1])

        ax_cm = self.figure.add_subplot(gs[:, 0])
        ax_loss = self.figure.add_subplot(gs[0, 1])
        ax_acc = self.figure.add_subplot(gs[1, 1])

        # ---- confusion matrix heatmap ----
        labels = [textwrap.fill(c, 12) for c in nn.class_names]
        n = len(labels)
        im = ax_cm.imshow(conf_matrix, cmap="Blues")
        self.figure.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

        ax_cm.set_xticks(range(n))
        ax_cm.set_yticks(range(n))
        ax_cm.set_xticklabels(labels, fontsize=8)
        ax_cm.set_yticklabels(labels, fontsize=8)
        ax_cm.set_xlabel("Predicted class")
        ax_cm.set_ylabel("Actual class")
        ax_cm.set_title(f"Test Confusion Matrix\nOverall accuracy: {overall:.2f}%",
                        fontsize=10)

        threshold = conf_matrix.max() / 2 if conf_matrix.max() else 0
        for i in range(n):
            for j in range(n):
                ax_cm.text(j, i, str(conf_matrix[i, j]),
                           ha="center", va="center", fontsize=14,
                           fontweight="bold",
                           color="white" if conf_matrix[i, j] > threshold
                           else "black")

        # ---- training curves ----
        epochs_axis = range(1, len(history["loss"]) + 1)

        ax_loss.plot(epochs_axis, history["loss"], color="tab:red")
        ax_loss.set_title("Training loss (MSE per epoch)", fontsize=10)
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.grid(alpha=0.3)

        ax_acc.plot(epochs_axis, history["accuracy"], color="tab:green")
        ax_acc.set_title("Training accuracy per epoch", fontsize=10)
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_ylim(0, 105)
        ax_acc.grid(alpha=0.3)

        self.figure.suptitle(title_extra, fontsize=9, color="#444")
        self.canvas.draw()

    def _save_figure(self):
        if self.model is None:
            messagebox.showwarning("Nothing to save", "Train the network first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf")],
            initialfile=f"run_{self.run_count}_results.png")
        if path:
            self.figure.savefig(path, dpi=200)
            self.status_var.set(f"Saved: {path}")

    # ==================================================================
    # Actions
    # ==================================================================
    def _read_params(self):
        """Read + validate all user inputs. Raises ValueError on problems."""
        try:
            n_layers = int(self.layers_var.get())
        except ValueError:
            raise ValueError("Number of hidden layers must be an integer.")
        if not 1 <= n_layers <= 10:
            raise ValueError("Number of hidden layers must be between 1 and 10.")

        hidden = []
        for i, var in enumerate(self.neuron_vars, start=1):
            try:
                n = int(var.get())
            except ValueError:
                raise ValueError(f"Neurons in layer {i} must be an integer.")
            if n < 1:
                raise ValueError(f"Neurons in layer {i} must be at least 1.")
            hidden.append(n)

        try:
            eta = float(self.eta_var.get())
        except ValueError:
            raise ValueError("Learning rate must be a number.")
        if eta <= 0:
            raise ValueError("Learning rate must be greater than 0.")

        try:
            epochs = int(self.epochs_var.get())
        except ValueError:
            raise ValueError("Number of epochs must be an integer.")
        if epochs < 1:
            raise ValueError("Number of epochs must be at least 1.")

        return hidden, eta, epochs

    def _train(self):
        try:
            hidden, eta, epochs = self._read_params()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return

        activation = self.act_var.get()
        use_bias = self.bias_var.get()

        self.train_btn.config(state="disabled")
        self.status_var.set("Training... (progress is printed in the console)")
        self.update_idletasks()

        try:
            self.run_count += 1
            print(f"\n===== Run {self.run_count} =====")

            self.model, train_acc, test_acc, conf_matrix, confusion_df = \
                nn.run_model(hidden, eta, epochs, activation, use_bias)
        except Exception as e:
            self.run_count -= 1
            self.status_var.set("Error.")
            messagebox.showerror("Error", f"{type(e).__name__}: {e}")
            return
        finally:
            self.train_btn.config(state="normal")

        overall = conf_matrix.trace() / conf_matrix.sum() * 100
        architecture = [nn.X_train.shape[1]] + hidden + [len(nn.class_names)]
        details = (f"Run {self.run_count} | Architecture {architecture} | "
                   f"{activation} | bias: {'yes' if use_bias else 'no'} | "
                   f"eta {eta} | {epochs} epochs")

        # also print to the console for each run
        print(details)
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Testing Accuracy: {overall:.2f}%")
        print("Test Confusion Matrix:")
        print(confusion_df)

        # visual results in the GUI
        self.summary_var.set(f"Training accuracy: {train_acc:.2f}%     "
                             f"Testing accuracy: {overall:.2f}%")
        self._draw_results(conf_matrix, self.model.history, overall, details)

        self.status_var.set(f"Run {self.run_count} finished.")
        self.class_result_var.set("Network trained. Enter a sample and classify.")
        self.tabs.select(0)

    def _classify(self):
        if self.model is None:
            messagebox.showwarning("Not trained", "Train the network first.")
            return

        try:
            values = [float(v.get()) for v in self.sample_vars.values()]
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "All numeric features must be numbers.")
            return

        location = self.location_var.get()
        if not location:
            messagebox.showerror("Invalid input", "Choose an origin location.")
            return

        try:
            name, output = nn.classify_sample(self.model, values + [location])
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        scores = "\n".join(
            f"   {c}: {o:.4f}" for c, o in zip(nn.class_names, output))
        self.class_result_var.set(f"Predicted class: {name}\n\nOutputs:\n{scores}")


if __name__ == "__main__":
    App().mainloop()
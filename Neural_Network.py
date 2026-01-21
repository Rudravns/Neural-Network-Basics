import numpy as np
import helper
import load_data
from typing import Tuple
from scipy.ndimage import zoom


class Neural_Network:
    def __init__(self, Input_layer_size: int, Output_layer_size: int, Hidden_Layers_Sizes: Tuple[int, ...]):
        self.Input_layer_size = Input_layer_size
        self.Output_layer_size = Output_layer_size
        self.Hidden_Layers_Sizes = Hidden_Layers_Sizes

        self.layer_sizes = [Input_layer_size, *Hidden_Layers_Sizes, Output_layer_size]

        # Initialize weights and biases
        self.weights, self.biases = self.__init_weights()

    # --------------------------------------------------
    # Forward pass
    # --------------------------------------------------
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.activations = [inputs]  # store activations
        self.z_values = []           # store pre-activations for ReLU derivative
        activation = inputs

        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = W @ activation + b
            self.z_values.append(z)
            if i < len(self.weights) - 1:  # hidden layers
                activation = self.__relu(z)
            else:  # output layer
                activation = self.__soft_max(z)
            self.activations.append(activation)

        return activation

    def start(self, *inputs: np.ndarray):
        return self.forward(np.array(inputs))

    def get_all_nodes(self):
        return self.Input_layer_size, self.Hidden_Layers_Sizes, self.Output_layer_size

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def __init_weights(self):
        data = helper.load()
        if data != {}:
            # Validate loaded weights match current architecture
            weights = data["weights"]
            biases = data["biases"]
            valid = True
            if len(weights) != len(self.layer_sizes) - 1:
                valid = False
            else:
                for i, w in enumerate(weights):
                    if w.shape != (self.layer_sizes[i+1], self.layer_sizes[i]):
                        valid = False
                        break
            if valid:
                # Check for corrupted weights (NaNs)
                if np.isnan(np.sum(weights[0])):
                    print("Corrupted weights detected. Re-initializing.")
                    return self.__random_weights()
                return weights, biases
            print("Architecture mismatch. Re-initializing weights.")

        weights = []
        biases = []

        for i in range(len(self.layer_sizes) - 1):
            self.__create_layer(weights, biases, i)

        return weights, biases

    def __random_weights(self):
        weights = []
        biases = []
        for i in range(len(self.layer_sizes) - 1):
            self.__create_layer(weights, biases, i)
        return weights, biases

    def __create_layer(self, weights, biases, i):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            limit = np.sqrt(2 / n_in)  
            W = np.random.uniform(-limit, limit, (n_out, n_in))
            b = np.zeros(n_out)
            weights.append(W)
            biases.append(b)

    def __relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def __soft_max(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=0, keepdims=True)


class Neural_Network_Trainer:
    def __init__(self, nn: Neural_Network):
        self.nn = nn

        load_data.extract_mnist()
        self.training_images = load_data.load_mnist_images("train-images-idx3-ubyte")
        self.training_labels = load_data.load_mnist_labels("train-labels-idx1-ubyte")
        self.test_images = load_data.load_mnist_images("t10k-images-idx3-ubyte")
        self.test_labels = load_data.load_mnist_labels("t10k-labels-idx1-ubyte")

    def __augment(self, img: np.ndarray) -> np.ndarray:
        # ------------------------------
        # Random shift (-2 to +2 pixels)
        # ------------------------------
        dx = np.random.randint(-2, 3)
        dy = np.random.randint(-2, 3)

        shifted = np.roll(img, dy, axis=0)
        shifted = np.roll(shifted, dx, axis=1)

        # Fix roll artifacts
        if dy > 0: shifted[:dy, :] = 0
        elif dy < 0: shifted[dy:, :] = 0
        if dx > 0: shifted[:, :dx] = 0
        elif dx < 0: shifted[:, dx:] = 0

        # ------------------------------
        # Normalize
        # ------------------------------
        x = shifted.astype(np.float32) / 255.0

        # ------------------------------
        # Random scale (big / small)
        # ------------------------------
        scale = np.random.uniform(0.8, 1.2)  # <1 smaller, >1 bigger
        x_scaled = zoom(x, scale, order=1)

        # ------------------------------
        # Center crop or pad to 28x28
        # ------------------------------
        shape = x_scaled.shape
        h: int = int(shape[0])  # type: ignore
        w: int = int(shape[1]) if len(shape) > 1 else 1  # type: ignore
        out = np.zeros((28, 28), dtype=np.float32)

        # Crop if too big
        if h > 28:
            top = (h - 28) // 2
            x_scaled = x_scaled[top:top + 28, :]
        if w > 28:
            left = (w - 28) // 2
            x_scaled = x_scaled[:, left:left + 28]

        # Pad if too small
        shape = x_scaled.shape
        h = int(shape[0])  # type: ignore
        w = int(shape[1]) if len(shape) > 1 else 1  # type: ignore
        out[(28 - h) // 2:(28 - h) // 2 + h,
            (28 - w) // 2:(28 - w) // 2 + w] = x_scaled

        # ------------------------------
        # Add noise
        # ------------------------------
        noise = np.random.normal(0, 0.05, out.shape)
        out += noise

        return np.clip(out, 0, 1).flatten()

    @helper.time_it
    def train(self, epochs: int, batch_size: int, learning_rate: float):
        print("=====================================")
        print("       Neural Network Learning       ")
        print("=====================================")

        for epoch in range(epochs):
            permutation = np.random.permutation(len(self.training_images))
            shuffled_images = self.training_images[permutation]
            shuffled_labels = self.training_labels[permutation]

            for i in range(0, len(shuffled_images), batch_size):
                batch_images = shuffled_images[i:i + batch_size]
                batch_labels = shuffled_labels[i:i + batch_size]

                # Accumulate gradients for the batch
                grad_W = [np.zeros_like(w) for w in self.nn.weights]
                grad_b = [np.zeros_like(b) for b in self.nn.biases]

                for img, label in zip(batch_images, batch_labels):
                    x = self.__augment(img)
                    y_true = np.zeros(self.nn.Output_layer_size)
                    y_true[label] = 1

                    # forward
                    y_pred = self.nn.forward(x)

                    # output error
                    delta = y_pred - y_true

                    # backprop
                    for l in reversed(range(len(self.nn.weights))):
                        a_prev = self.nn.activations[l]
                        dW = np.outer(delta, a_prev)
                        db = delta

                        # gradient clipping
                        dW = np.clip(dW, -1, 1)
                        db = np.clip(db, -1, 1)

                        grad_W[l] += dW
                        grad_b[l] += db

                        # Calculate delta for previous layer
                        if l > 0:
                            W = self.nn.weights[l]
                            z_prev = self.nn.z_values[l - 1]
                            new_delta = (W.T @ delta) * (z_prev > 0)
                            delta = new_delta

                # Update weights after batch (Average gradients)
                batch_len = len(batch_images)
                for l in range(len(self.nn.weights)):
                    self.nn.weights[l] -= (learning_rate / batch_len) * grad_W[l]
                    self.nn.biases[l] -= (learning_rate / batch_len) * grad_b[l]


            print(f"Epoch {epoch + 1}/{epochs} completed.")

        helper.save({"weights": self.nn.weights, "biases": self.nn.biases})

    def train_manual(self, img: np.ndarray, label: int, learning_rate: float = 0.1):
        # Train on this specific image multiple times with augmentation
        # to help the network generalize and learn it firmly.
        # Create a mixed batch of the new manual image AND random old images.
        # This prevents "Catastrophic Forgetting" (where learning 10 makes it forget 0-9).

        samples = []

        # 1. Add the manual image (we'll augment it 5 times)
        for _ in range(5):
            x = self.__augment(img)
            samples.append((img, label))

        # 2. Add random existing images (Replay Buffer)
        # We mix in ~25 old images to remind the AI what 0-9 look like.
        # This ensures it learns "This is 10" AND "These others are NOT 10".
        num_replays = 25
        indices = np.random.randint(0, len(self.training_images), num_replays)
        for idx in indices:
            samples.append((self.training_images[idx], self.training_labels[idx]))

        # Shuffle so we don't train all 10s in a row
        np.random.shuffle(samples)

        for x_img, y_label in samples:
            x = self.__augment(x_img)
            y_true = np.zeros(self.nn.Output_layer_size)
            y_true[label] = 1
            y_true[y_label] = 1

            # Forward
            y_pred = self.nn.forward(x)

            # Error
            delta = y_pred - y_true

            # Backprop (Stochastic Gradient Descent)
            for l in reversed(range(len(self.nn.weights))):
                a_prev = self.nn.activations[l]
                dW = np.outer(delta, a_prev)
                db = delta

                # Gradient clipping
                dW = np.clip(dW, -1, 1)
                db = np.clip(db, -1, 1)

                # Calculate delta for previous layer BEFORE update
                if l > 0:
                    W = self.nn.weights[l]
                    z_prev = self.nn.z_values[l - 1]
                    new_delta = (W.T @ delta) * (z_prev > 0)

                # Update weights
                self.nn.weights[l] -= learning_rate * dW
                self.nn.biases[l] -= learning_rate * db

                if l > 0:
                    delta = new_delta

    def test(self):
        print("\n\n=====================================")
        print("        Neural Network Test          ")
        print("=====================================")
        correct = 0

        for img, label in zip(self.test_images, self.test_labels):
            x = img.flatten() / 255.0
            y_pred = self.nn.forward(x)
            predicted_label = np.argmax(y_pred)
            if predicted_label == label:
                correct += 1

            print(f"Predicted: {predicted_label}, Actual: {label}")


        accuracy = correct / len(self.test_images) * 100
        print(f"Test accuracy: {accuracy:.2f}%")
        return accuracy
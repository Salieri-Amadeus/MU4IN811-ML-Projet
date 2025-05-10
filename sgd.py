import numpy as np

def SGD(optim, X, Y, batch_size=32, epochs=1000, verbose=True):
    N = X.shape[0]
    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        Y_shuffled = Y[indices]

        epoch_loss = 0
        correct = 0

        for i in range(0, N, batch_size):
            x_batch = X_shuffled[i:i+batch_size]
            y_batch = Y_shuffled[i:i+batch_size]

            loss = optim.step(x_batch, y_batch)
            epoch_loss += loss * len(x_batch)

            y_pred = (optim.net.forward(x_batch) >= 0.5).astype(int)
            correct += (y_pred == y_batch).sum()

        epoch_loss /= N
        acc = correct / N
        history["loss"].append(epoch_loss)
        history["accuracy"].append(acc)

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {epoch_loss:.4f} | Acc = {acc:.2%}")

    return history

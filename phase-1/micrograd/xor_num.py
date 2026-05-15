import numpy as np

# --- SETUP: THE FACTORY BLUEPRINT ---
# Front Door: XOR Truth Table
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0],    [1],    [1],    [0]])

np.random.seed(100) # Keep results consistent
lr = 0.45
epochs = 100000 # Enough to see loss drop clearly

# Building the Rooms: Random Weights & Biases
# Room 1 (Hidden): 2 inputs -> 2 neurons
w1 = np.random.uniform(size=(2, 2))
b1 = np.zeros((1, 2))

# Room 2 (Output): 2 hidden neurons -> 1 output
w2 = np.random.uniform(size=(2, 1))
b2 = np.zeros((1, 1))

# The Toll Booth Functions
def sigmoid(val):
    return 1 / (1 + np.exp(-val))

def sigmoid_derivative(activated_val):
    # This assumes 'activated_val' is already passed through sigmoid
    return activated_val * (1 - activated_val)

# --- THE TRAINING LOOP ---
for i in range(epochs):
    
    # PHASE 1: THE ASSEMBLY LINE (Forward Pass)
    # Entering Room 1
    z1 = np.dot(x, w1) + b1
    a1 = sigmoid(z1) # Hidden signal (Squashed)
    
    # Entering Room 2
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2) # Final Prediction (Squashed)

    # PHASE 2: THE "OOPS" FACTOR
    error = y - a2

    # PHASE 3: THE DEMON'S WORK (Backward Pass)
    # Room 2's Blame (Equation BP1)
    # Multiply the "Oops" by how "Awake" Room 2 is
    grad_out = error * sigmoid_derivative(a2)

    # Passing the Blame backwards through the hallways (Equation BP2 part 1)
    error_a1 = np.dot(grad_out, w2.T)

    # Room 1's Blame (Equation BP2 part 2)
    # Multiply the incoming blame by how "Awake" Room 1 is
    grad_hidden = error_a1 * sigmoid_derivative(a1)

    # PHASE 4: TURNING THE KNOBS (Weight Updates)
    # Fixing Room 2 (Equation BP4: a_in * grad_out)
    w2 += np.dot(a1.T, grad_out) * lr
    b2 += np.sum(grad_out, axis=0, keepdims=True) * lr

    # Fixing Room 1 (Equation BP4: input_x * grad_hidden)
    w1 += np.dot(x.T, grad_hidden) * lr
    b1 += np.sum(grad_hidden, axis=0, keepdims=True) * lr

    # Print status every 10k epochs
    if i % 10000 == 0:
        loss = np.mean(np.square(error))
        print(f"Epoch {i} | Loss: {loss:.6f}")

# --- FINAL TEST: CHECKING THE RESULTS ---
print("\n--- Final XOR Predictions ---")
for idx in range(len(x)):
    # Final forward pass for testing
    test_z1 = np.dot(x[idx], w1) + b1
    test_a1 = sigmoid(test_z1)
    test_z2 = np.dot(test_a1, w2) + b2
    test_a2 = sigmoid(test_z2)
    
    print(f"Input: {x[idx]} | Target: {y[idx]} | Prediction: {test_a2[0][0]:.4f}")
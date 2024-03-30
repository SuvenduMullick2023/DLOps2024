import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
  """Sigmoid activation function"""
  return 1 / (1 + np.exp(-x))

def relu(x):
  """ReLU activation function"""
  return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
  """Leaky ReLU activation function with alpha parameter"""
  return np.maximum(alpha * x, x)

def tanh(x):
  """Tanh activation function"""
  return np.tanh(x)

# Define input values
#x = np.linspace(-5, 5, 100)  # Range of input values
random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
#x = np.array(random_values)
#x =np.sort(x) 
# Generate activation function outputs
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_tanh = tanh(x)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each activation function with a label
plt.plot(x, y_sigmoid, label='Sigmoid')
plt.plot(x, y_relu, label='ReLU')
plt.plot(x, y_leaky_relu, label='Leaky ReLU')
plt.plot(x, y_tanh, label='Tanh')

# Add labels and title
plt.xlabel('Input (x)')
plt.ylabel('Output (y)')
plt.title('Activation Functions')

# Add legend
plt.legend()

# Set grid
plt.grid(True)

# Show the plot
plt.show()

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate synthetic data for demonstration
# Replace this with your actual dataset
num_samples = 1000
backspace_counts = np.random.randint(0, 50, size=num_samples)
prompt_lengths = np.random.randint(5, 50, size=num_samples)
labels = np.random.randint(0, 2, size=num_samples)

# Normalize the features
backspace_counts_norm = backspace_counts / 50  # Max value of backspace count is 50
prompt_lengths_norm = prompt_lengths / 50  # Max value of prompt length is 50

# Combine features into input array
X = np.column_stack((backspace_counts_norm, prompt_lengths_norm))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

import numpy as np

# Actions/ sign language gestures to detect
gestures = np.array([
    "None",         # No gesture
    "Hello",        # Greeting gesture
    "Good",         # Gesture indicating something good
    "Thank you",    # Gesture expressing gratitude
    "Sorry",        # Gesture for apology
    "How are you?", # Gesture to ask about well-being
    "Night",        # Gesture for saying goodnight
    "Morning"       # Gesture for saying good morning
])

# Number of sequences per gesture for training
no_sequences = 30

# Length of each sequence of gestures
sequence_length = 30

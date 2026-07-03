"""Global configuration for the DDQN text-hint ITS prototype."""

# State and action dimensions
STATE_DIM = 6
ACTION_DIM = 3

# Double DQN hyperparameters
LEARNING_RATE = 0.001
GAMMA = 0.99
BATCH_SIZE = 64
BUFFER_SIZE = 10000
NUM_EPISODES = 500
MAX_STEPS_PER_EPISODE = 50

# Dynamic epsilon-greedy exploration
EPSILON_MIN = 0.05
EPSILON_MAX = 0.80
LAMBDA_ENTROPY = 0.50
SOFTMAX_TEMPERATURE = 1.0

# Target network update frequency
TARGET_UPDATE_FREQUENCY = 20

# Reward weights used by the tutoring environment
REWARD_WEIGHTS = {
    "correct": 1.0,
    "improvement": 0.3,
    "efficient_response_time": 0.2,
    "extra_attempt_penalty": -0.1,
    "hint_usage_penalty": -0.1,
    "unnecessary_strong_hint_penalty": -0.1,
    "incorrect_penalty": -0.5,
}

# Environment normalization constants
MAX_RESPONSE_TIME = 120.0
MAX_ATTEMPTS = 3

# Fixed action semantics
ACTION_STRONG_HINT = 0
ACTION_MEDIUM_HINT = 1
ACTION_WEAK_HINT = 2

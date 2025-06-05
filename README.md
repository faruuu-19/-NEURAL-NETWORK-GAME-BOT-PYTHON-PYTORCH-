# -NEURAL-NETWORK-GAME-BOT-PYTHON-PYTORCH-
 Designed and trained an AI agent to play a 2D grid-based game  using feedforward neural networks and reward-based feedback  loops. Implemented state-action encoding, loss tracking, and real time performance visualization.
# Street Fighter II AI Bot

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-ML-green)](https://www.tensorflow.org/)
This project implements a machine learning–based bot that plays
 **Street Fighter II**. Using a provided emulator and API, the bot interacts with the game in real time and replaces hardcoded rules with a learned policy that adapts to different characters and scenarios.

## Project Overview

- **Language**: Python  
- **Framework**: TensorFlow  
- **Goal**: Predict button presses based on current game state using a trained model

## Machine Learning Pipeline

### 1. Game State as Input

Each frame of gameplay is treated as a training instance. Features are extracted from the `GameState` object, including:

- Player and opponent health
- X and Y positions
- Current moves
- Active button states

### 2. Dataset Construction

- Data was collected during manual single-player gameplay sessions  
- Duplicate rows were removed  
- Irrelevant columns were dropped  
- Final dataset saved as a CSV file  
  - Each row represents a frame of gameplay  
  - The last 10 columns represent button states: Up, Down, Left, Right, A, B, X, Y, L, R

### 3. Model Architecture

A **Multilayer Perceptron (MLP)** model was built using TensorFlow:

- Input layer: accepts all preprocessed features  
- 4 hidden layers: each with ReLU activation  
- Output layer: 10 sigmoid units, one for each button

### 4. Output Strategy

- The model outputs a 10-dimensional binary vector  
- Each output represents whether a button should be pressed  
- Sigmoid activation allows each button to be treated independently

### 5. Data Cleaning Improvements

Additional observations during gameplay helped refine the dataset:

- Better handling of idle states and corner-specific behaviors  
- Improved model performance through more robust input representation

## Bot Integration

After training, the model is used in the game bot:

- `bot.py` loads the trained model  
- For each game frame:  
  - GameState is parsed  
  - Features are passed to the model  
  - The model outputs predictions  
  - A `Command` object is created from the predicted button states  
- The bot presses the predicted buttons in real time

## Results and Observations

- The bot successfully connected with the emulator and played matches using learned behavior  
- Performance improved with:  
  - Larger, cleaner datasets  
  - Improved handling of edge cases  
- Demonstrated responsive and adaptive behavior in real time


## Setup Instructions

### Requirements

- Python 3.7+  
- TensorFlow  
- NumPy  
- pandas

### Installation

```bash
git clone https://github.com/yourusername/sf2-ml-bot.git
cd sf2-ml-bot
pip install -r requirements.txt



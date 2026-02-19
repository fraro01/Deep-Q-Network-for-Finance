# Short-Term Investment Strategies with Deep Q-Network (DQN)

This repository contains my **Applied Reinforcement Learning** exam project: a **Deep Q-Network (DQN)** agent trained to perform **short-term trading** on historical market data.

The core idea is to model a simplified trading task as a **Gymnasium-compatible environment**, then train a **value-based RL agent** (DQN) to learn a policy over three discrete actions: **Buy / Hold / Sell**.

<p align="center">
  <img src="trading_bot.JPG" width="50%" alt="trading bot">
  <br>
  <em> Ideal representation of a trading bot </em>
</p>

---

## Project Overview

### What this project does
- Builds a custom **Gymnasium environment** (`TradingEnv`) using daily prices downloaded via `yfinance`.
- Represents the environment **state** as a **sliding window of percentage price changes**.
- Trains a **DQN agent** in PyTorch using:
  - replay memory
  - target network updates
  - epsilon-greedy exploration
- Produces a simple **rendering dashboard** with Matplotlib showing price history and the agent’s actions.

### Action space
- `0`: Buy (1 share if affordable)
- `1`: Hold
- `2`: Sell (1 share if held)

### Reward
The reward is a **monetary reward**, computed as the variation of portfolio value between consecutive steps:
- Portfolio value = `cash + shares * price`
- Reward = `new_portfolio_value - old_portfolio_value`

This aligns the RL objective with the trading objective: **maximize final portfolio value**.

---

## Repository Structure

- `project.ipynb`  
  Main notebook: methodology, experiments, training loop, results and plots.

- `tradingenv.py`  
  Custom Gymnasium environment implementation (`TradingEnv`).

- `requirements.txt`  
  Python dependencies.

- `trading_bot.JPG`, `plot.JPG`  
  Images used in the README.

---

## Requirements

- Python (version specified in `requirements.txt`)
- Dependencies listed in `requirements.txt`

---

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
2. Move into the project folder:
   ```bash
   cd <folder>
3. Install dependencies:
   ```bash
   pip install -r requirements.txt

---

## How to run
Open the notebook and execute the cells in order:
   ```bash
   jupyter notebook project.ipynb
   ```

Inside the notebook you can configure:

* ticker symbol (e.g., SPY, AAPL, QQQ)
* date range
* data granularity (daily)
* sliding window length
* DQN hyperparameters (gamma, learning rate, batch size, epsilon schedule, ecc...)

---

Dashboard Preview
<p align="center"> <img src="plot.JPG" width="66%" alt="Rendering of the taken actions"> <br> <em>Example of price series with the agent’s Buy/Sell actions</em> </p>

---

## Notes & Limitations

This project is an educational implementation and intentionally simplifies several real-world aspects of trading:

* No transaction costs / slippage (unless explicitly enabled in code).

* The market dynamics are exogenous: the agent’s actions do not influence future prices.

* Generalization is evaluated via out-of-sample testing (recommended), since in-sample performance can be misleading.

If you want to extend the project, good next steps are:

* transaction costs and spread modeling

* more informative state features (technical indicators, volatility measures, etc.)

* walk-forward validation

* Double DQN / Dueling DQN / Prioritized Experience Replay

---

## References / Credits
* Professor Berta’s RL material and examples:
   https://github.com/riccardoberta/reinforcement-learning

* q-trader (inspiration for trading-RL framing):
   https://github.com/edwardhdlu/q-trader

* Gymnasium (Farama Foundation):
   https://gymnasium.farama.org/

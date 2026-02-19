# Short-Term Investment Strategies with Deep Q-Network (DQN)

This repository contains the `Jupyter Notebook` project of the Applied Reinforcement Learning exam.

The aim of this project is the application of **Deep Reinforcement Learning**, and in particular the **Deep Q-Network (DQN)** algorithm in the world of finance, in order to train an "intelligent agent" to perform successfuly short-term investments.

<p align="center">
   <img src="trading_bot.JPG" width="50%" alt="trading bot">
   <figcaption align="center">Ideal representation of a trading bot</figcaption> 
</p>

The project is composed by two fundamental files:

  * the `Jupyter Notebook` file, containing the development of the project with detailed explanations an motivations of all the resonings.
  * the Python script: `tradingenv.py`, fundamental for the correct run of the Notebook, since it contains the creation of the environment.
  * 
The environment has been created ad-hoc environment, such that inherits from the Gymnasium base environment class, hance it has all the main methods.\
This is done in order to stick to the interoperability of our architecture as much as possible.

The DQN architecture is implemented using the **[Pytorch](https://pytorch.org/)** framework and rendered as a dashboard using **[Matplotlib](https://matplotlib.org/)**.

---

## Requirements

- Python (version as specified in `requirements.txt`)
- Required Python libraries (see `requirements.txt`)

---

## Installation

1. Download the folder manually or clone the repository:
   ```bash
   git clone <repository-url>

2. Navigate to the local folder:
   ```bash
   cd folder
  
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Make sure to have Python installed, and `pip` working properly from `cmd`.

---

## Dashboard Preview
<p align="center"> <img src="plot.JPG" width="66%" alt="Rendering of the taken actions"> </p>

---

## Citation

> https://github.com/riccardoberta/reinforcement-learning  
> https://github.com/edwardhdlu/q-trader  
> [Gymnasium, Farama Foundation](https://gymnasium.farama.org/index.html)

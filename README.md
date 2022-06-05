inspiration# Machine Learning for a Mechanical ventilator
## Overview
this project is a [Kaggle challenge](https://www.kaggle.com/competitions/ventilator-pressure-prediction) and it aims to develop a ML model to predict the lung pressure in a Mechanical ventilator.  <br>

In times of the global pandemic caused by COVID19, medical services experience an exceeding workload caused by the large number of patients in the hospitals. A significant part of these patients needs mechanical ventilation systems (MVS) occupying additional trained medical personnel. These specially trained people need to adjust the lung pressure on the MVS manually to ensure optimal oxygen supply. At the same time they need to prevent severe lung damage due to high pressure. <br> 
Also developing new methods for controlling mechanical ventilators is prohibitively expensive, even before reaching clinical trials. High-quality simulators could reduce this barrier. Current simulators are trained as an ensemble, where each model simulates a single lung setting. However, lungs and their attributes form a continuous space, so a parametric approach must be explored that would consider the differences in patient lungs.  <br>

A fully automatic MVS would reduce the workload of the medical staff and enable better care for other patients. For the adaptive and automatic mode of operation of such a device, machine learning (ML) methods are required. Our project aims to predict the lung pressure based on different lung attributes using various ML approaches. In that way, important parameters of the MVS can be assessed that help to build a superior software that assist to find an pressure output for the individual patient. By combining flexible models with time series elements, we are able to make predictions with high precision.

## Stakeholder info
Our stakeholder is People's Ventilator project(PVP), a project funded by Princeton University, as a team at Google Brain aims to grow the community around machine learning for mechanical ventilation control. They believe that neural networks and deep learning can better generalize across lungs with varying characteristics than the current industry standard of PID controllers. They came up with a mechanical ventilator which can be build in 3 days by a single person and it only costs around 1300$ (current existing ventilators: <10,000$). At the end the aim is to make it available for different health sectors all over the world!

## Data Structure
We were given different breaths corrsponding to different lung settings and the task was to predict the airway pressure in the respiratory circuit during the breath, given the time series of control inputs. Basic data structure looks like: <br> 
time_step : 80 timesteps corresponding to a single breath <br> 
u_in : Input flow which correspond to the amount of air that is flowing into the lungs <br> 
u_out : Indicator variable which tells whether the expiratory valve is open or not. If open, denoted by 0 and if closed (inspiration), denoted by  1 (exspiration) <br> 
R (Resistance) : Continuity the airway system of a lung <br>
C (Compliance) : Flexibility of the lungs and how it corresponds to the input flow <br>
Pressure : Target variable. Lung pressure during different phases of the breathing cycles.

Apart from this basic features, we calculated additional feature doing feature engineering for e.g.: different air volumns and time shifted features.

## Evaluation metrics
It is very important for us to predict the target variable, i.e. pressure very accurately. Because, if it is too low, the patient won't get enough oxygen and if it is too high, it can cause lung damage. Hence we need to have a measure of error corresponding to the pressure. In this case, we choose 
mean absolue error (MAE) as our error metric since it can give us the mean deviation of our prediction from the actual values of the target variable very well, in our case pressure. MAE is too sensitive to lower and higher values of pressure.

## Baseline model
We choose our baseline model to be a linear regression with polynomial function (degree 2) on our basic features (input flow, output flow, R and C). 
Results are shown in the notebook: [baseline_model.ipynb](https://github.com/Lue-C/CapStone/blob/main/models/baseline_model.ipynb). This model gave us an MAE of 3.38 cmH2O and it clearly showed that our model is not enough to cope with very different pressure profiles. So we needed more complex models.

## Machine Learning models
Complex models like XGBoost and Artificial Neural Network (ANN) gave us promising result in the initial stages. But then we realised that this problem can be 
treated as a auto regression (AR) problem since the pressure values at a given time highly depends on the pressure values at the previous time steps. Hence we calculated AR features out of given features (pressure and input flow) by time shifting them. We used these features along with the other feature engineed in XGBoost and ANN. As we expected, the results were promising and ANN gave us the best results so far with an MAE of 0.15 ([ANN.ipynb](https://github.com/Lue-C/CapStone/blob/main/models/ANN.ipynb))

# Installation
## Requirements

- pyenv with Python: 3.9.4

### Setup

Use the requirements file in this repo to create a new environment.

```BASH
make setup

#or

pyenv local 3.9.4
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_dev.txt
```

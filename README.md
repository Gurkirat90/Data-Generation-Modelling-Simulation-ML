# Warehouse Throughput : Simulation-to-ML Pipeline

**BY**: Gurkirat Singh  
**Roll Number**: 102303256  
**Group**: 3C21  

This project demonstrates a full-cycle data science workflow: building a **Discrete Event Simulation (DES)** to generate a synthetic dataset and evaluating multiple **Machine Learning** models to predict system performance.

---

##  Methodology

**Simulation Architecture**

Using SimPy, I developed a stochastic model of a warehouse fulfillment center. The simulation mimics the interaction between incoming orders and available robotic resources.
  -	Process: Orders arrive via an Exponential distribution. They enter a queue and wait for the next available robot.
-	Stochasticity: To ensure a realistic dataset, I introduced Gaussian noise into the processing times and a 5% "system glitch" probability to simulate real-world variance.

```bash
!pip install simpy
```

**Data Generation**

Executed **1,000 independent simulation** runseach representing an 8-hour work shift. For each run, I randomized three key input parameters (features) and recorded the resulting system performance (target).


**Dataset Parameters**

| Parameter        | Type    | Range / Value | Description |
|------------------|---------|---------------|-------------|
| Num Robots       | Feature | 1 – 20        | Total number of robots available to process tasks. |
| Arrival Rate     | Feature | 2.0 – 15.0    | Mean time (in minutes) between incoming orders. |
| Process Time     | Feature | 3.0 – 12.0    | Mean time (in minutes) a robot takes to complete a task. |
| Avg Wait Time    | Target  | Measured      | Average time an order spends waiting in the queue. |


**Dataset.head()**


```
num_robots  arrival_rate  process_time  avg_wait_time
0          20      8.636156     11.128049      11.018588
1           6     13.364032      7.530694       7.809437
2          16     14.282148      5.010451       5.318448
3           4     11.767669      5.720350       5.688438
4          20      5.939002      7.971642       7.937993

```


**Dataset Description**


```
num_robots  arrival_rate  process_time  avg_wait_time
count  1000.000000   1000.000000   1000.000000    1000.000000
mean     10.203000      8.443807      7.455073       7.664041
std       5.660551      3.799545      2.663872       2.911378
min       1.000000      2.003561      3.000953       2.651470
25%       5.000000      5.225459      4.973460       5.227559
50%      10.000000      8.456888      7.593287       7.686797
75%      15.000000     11.718577      9.806853       9.962766
max      20.000000     14.968261     11.986841      18.237425
```

---


##  Analysis

The goal was to determine which regression algorithm could best "reverse-engineer" the simulation's logic. I compared six different models using a 80/20 train-test split.


**ML Model Comparison**

| Model                | R² Score | MAE (mins) | RMSE |
|----------------------|----------|------------|------|
| Linear Regression    | 0.9000   | 0.51       | 0.96 |
| Gradient Boosting    | 0.8899   | 0.54       | 1.01 |
| Random Forest        | 0.8775   | 0.61       | 1.06 |
| K-Nearest Neighbors  | 0.8740   | 0.68       | 1.08 |
| XGBoost              | 0.8566   | 0.63       | 1.15 |
| Decision Tree        | 0.8021   | 0.71       | 1.35 |


**Why Linear Regression Won Contrary to the typical "more complex is better" rule ?**

Linear Regression outperformed ensemble methods like XGBoost.
  - Structural Alignment: The underlying physics of a queue-based simulation is largely additive. Linear Regression effectively captured the global trend without being distracted by the local noise added during the simulation.
  - Efficiency: The model achieved the highest accuracy ($R^2 = 0.90$) with the lowest computational overhead.


**Feature Importance**

The model identified **Num Robots** as the most significant predictor of wait times. The relationship is non-linear at the extremes **(the "bottleneck effect")**, but linear enough within our parameter bounds for the regression to maintain high precision.

---

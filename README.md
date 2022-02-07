# Volatility Prediction using ML

- Volatility Prediction using Voting Ensemble
  - Each Estimator's hyper parameters are optimized by Genetic Search
    - ElasticNet
    - SVR
    - RandomForest Regressor
    - XGBoost Regressor

- goal : predict next month's volatility and find max-sharpe weights of portfolio at the predicted volatility
- Train : 6 Months Close Price + Various technical indicators + Average Standard Deviation of portfolio
  - BWI
  - ATR
  - Relative Volatility Index
  - Standard Deviation
  - Sigma
  - True Range
- Target : next month volatility of portfolio

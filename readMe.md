**Forecasting Weekly Returns of Crude Oil Using Machine Learning and Macro-Financial Indicators: West Texas Intermediate**

**Project Overview** 

The main aim of this project is to forecast WTI returns using Macro-financial variables. The research questions are:

•	Primary question: How well can selected deep learning models forecast one-week-ahead WTI crude-oil returns using macro-financial variables? 

•	Sub-question: Will the selected models perform better than naïve benchmarks?

•	Sub-question: Which predictors heavily influence the nonlinear models?

•	Sub-question: Are the differences in accuracy between models statistically meaningful?



**Data** 

Initial data period: 2010–2024

Approximate number of observations: 3,957



| Variable Name          | Source        | Frequency | ID                  |

|------------------------|---------------|-----------|---------------------|

| WTI                    | FRED API      | Daily     | DCOILWTICO          |

| S\&P 500                | Yahoo Finance | Daily     | ^GSPC               |

| Volatility Index (VIX) | FRED API      | Daily     | VIXCLS              |

| USD                    | FRED API      | Daily     | DTWEXBGS            |

| Gold                   | Yahoo Finance | Daily     | GC=F                |

| Yield Difference       | FRED API      | Daily     | DGS3 and DGS10      |

| Oil Volatility (OVX)   | Yahoo Finance | Daily     | ^OVX                |

| Brent                  | FRED API      | Daily     | DCOILBRENTEU        |

| Industrial Production  | FRED API      | Monthly   | INDPRO              |



Wti\_data.csv file is the raw extracted data, while model\_data.csv is the data set used for modelling for the main regression analyses. 



**Structure** 

WTI\_Forecasting\_Artefact/

│

├── README.md

├── requirements.txt

│

├── data/

│   ├── model\_data/

│       └── model\_data.csv

│   └── raw\_data/

│       └── wti\_data.csv

│

├── code Files/

│   ├── oil\_prediction.ipynb

│

├── outputs/

│   ├── classification\_results.png

│   ├── daily\_regression\_results.png

│   ├──  Main\_results.png

│   ├──  Model\_roles.png

│   ├──  regime\_based\_results.png

│   ├──  sample\_model\_data.png

│   ├──  sample\_raw\_dataset.png

└── documentation/

&#x20;   ├── flowchart.png





**Methodology and Models**

Data set is converted into weekly frequencies and features are engineered using returns, differences and lags. A chronological 80/20 split train-test split is done. Hyperparameter tuning and Time series CV are used where appropriate. Classification, regime and daily experiments were carried out for additional analyses. The models used for the main reggresion study were: 

•	Mean, Zero and Persistence baselines 

•	Linear Regression 

•	Random Forest

•	XGBoost

•	Prophet

•	Sequence LSTM

Evaluations carried out were:

•	RMSE 

•	MAE

•	R-squared 

•	Directional Accuracy 

•	Diebold-Mariano Test

•	SHAP / feature importance 



**Some important notes**

Random seeds were used and no random shuffling was utilized. Tuning was carried out just on the training data.

How to run 

•	Install Python 3.11

•	Run “pip install -r requirements.txt”

•	Notebook should be run from top to bottom 

•	Main outputs are displayed at the bottom of each section 

**Repository Link**: https://github.com/Carlson29/Crude-Oil-Prices-Model.git

**Dependencies** 

pandas, numpy, fredapi, matplotlib, seaborn, yfinance, scikit-learn, scikit-optimize, statsmodels, scipy, xgboost, shap, tensorflow, prophet








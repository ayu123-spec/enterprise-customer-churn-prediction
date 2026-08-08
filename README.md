Enterprise Customer Churn Prediction

End-to-end machine learning service that scores telecom customers for churn risk, served as a REST API with a live web dashboard.

Live demo: https://enterprise-customer-churn-predictio-two.vercel.app API docs: https://customer-churn-prediction-hfwi.onrender.com/docs

Results

Both models were evaluated on a held-out stratified test set. Logistic regression is the deployed model.

Metric (churn class)	Logistic Regression	XGBoost
ROC-AUC	0.8417	0.8416
Precision	0.4992	0.5227
Recall	0.7914	0.7701
F1	0.6122	0.6227
Accuracy	0.7339	0.7523

Baseline: predicting "no churn" for every customer yields 73.5% accuracy and 0.500 ROC-AUC.

Lift over baseline: +0.342 ROC-AUC.

Reading these numbers honestly

The deployed model's accuracy (73.4%) sits marginally below the majority-class baseline (73.5%). This is not a defect — it is the expected consequence of the operating point chosen.

A model optimized for accuracy on a 26.5%-positive dataset learns to predict "no churn" almost always, catching nearly no churners while scoring well. This model instead catches 79% of customers who actually churn, accepting that roughly half of the accounts it flags would have stayed anyway. For a retention team, a missed churner costs an entire customer lifetime value, while a false positive costs one discount offer. The asymmetry justifies the trade.

ROC-AUC is the headline metric here because it measures ranking quality independent of threshold, and the ranking is what drives a prioritized outreach list.

Dataset
Property	Value
Source	IBM Telco Customer Churn
Rows	7,043 customers
Churn rate	26.5%
Features after engineering	36
Split	Stratified train/test
Serving performance
Metric	Value
Inference latency (p50, warm)	__ ms
Cold start (free tier, ~15 min idle)	30–60 s
Model artifact size	3.6 KB

The free Render instance sleeps when idle, so the first request after a pause pays a cold-start cost. This is a hosting trade-off chosen for a portfolio deployment, not a property of the model.

Model selection

The two models are statistically indistinguishable on ranking quality: 0.8417 versus 0.8416 ROC-AUC, a gap far smaller than test-set noise. XGBoost achieves slightly higher precision and accuracy; logistic regression achieves higher recall.

Logistic regression was deployed because it matches the ensemble's discriminative power at a 3.6 KB artifact with no additional dependency, and its coefficients are directly readable as feature effects, letting a retention team see which factors drive a given customer's risk score. When two models perform equivalently, the one that is smaller, faster to serve, and explainable to non-technical stakeholders is the better production choice.

Architecture
ingestion → preprocessing → feature engineering → training → evaluation
                                                       ↓
                            FastAPI service (Render) ← model artifact
                                       ↓
                        Static dashboard (Vercel), CORS-enabled
Stack

Python, scikit-learn, pandas, SHAP, FastAPI, uvicorn, joblib, pytest, Docker, Git. Frontend is dependency-free HTML/CSS/JS.
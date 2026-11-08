# telecom-customer-churn-analysis
End-to-end Telecom Customer Churn Analysis using Python, SQL, Machine Learning, and Power BI.
# Telecom Customer Churn Analysis

## Project Overview
An end-to-end data analytics project exploring customer churn in the telecommunications industry. The project uses Python, SQL, and Power BI to identify churn patterns, understand customer behaviour, and present actionable business insights.

## Business Problem
Customer churn affects recurring revenue and long-term business growth. This project investigates:
- What percentage of customers are leaving?
- Which contract types have higher churn?
- Which payment methods are associated with higher churn?
- How does customer tenure affect churn?
- Which customer groups may need retention attention?

## Tools & Technologies
- **Python:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- **SQL:** MySQL
- **Power BI:** Dashboard development, DAX measures, interactive slicers
- **Machine Learning:** Logistic Regression, Random Forest

## Dataset
IBM Telco Customer Churn dataset containing customer demographics, services, contract details, billing information, and churn status.

- Original records: 7,043
- Cleaned records: 7,032
- Target variable: `Churn`

## Project Workflow
1. Data cleaning and preprocessing using Python.
2. Exploratory Data Analysis (EDA) to identify customer churn patterns.
3. SQL analysis to calculate churn rates and compare customer segments.
4. Machine learning classification using Logistic Regression and Random Forest.
5. Interactive Power BI dashboard development.

## Key Business Insights

| Analysis | Finding |
|---|---:|
| Overall churn rate | 26.58% |
| Month-to-month contract churn | 42.71% |
| One-year contract churn | 11.28% |
| Two-year contract churn | 2.85% |
| Electronic check churn | 45.29% |
| 0–12 months tenure churn | 47.68% |
| Fiber optic churn | 41.89% |

Churned customers had an average monthly charge of $74.44, compared with $61.31 for customers who stayed.

These findings show associations in the dataset, not proof of causation.

## Power BI Dashboard

The interactive dashboard includes:
- Total customers
- Churned customers
- Overall churn rate
- Churn by contract type
- Churn by payment method
- Churn by tenure group
- Churn by internet service
- Interactive slicers

## Machine Learning
Classification models used:
- Logistic Regression
- Random Forest Classifier

Model evaluation and feature importance analysis were performed to explore churn prediction.

## Project Structure

telecom-customer-churn-analysis/
├── Python/
├── SQL/
├── powerbi/
├── Dataset/
├── Screenshots/
└── README.md

## Business Recommendations
- Investigate retention strategies for month-to-month customers.
- Review customer experience among electronic check users.
- Focus on early-tenure customer engagement.
- Investigate service-related concerns among fiber optic customers.

## Disclaimer
This project uses a publicly available IBM sample telecom churn dataset for educational and portfolio purposes. The findings should not be interpreted as results from an actual telecom company's private customer records.

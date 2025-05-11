"import openai 
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import re
import joblib


# Set your OpenAI API key
openai.api_key = "YOUR_API_KEY_HERE"  # Replace with your actual API key

# Load financial data
file_path = r"YOUR_DATA_HERE"
df = pd.read_csv(file_path, encoding="utf-8")

print(df.columns)

# Backup original stock prices
original_prices = df[["Instrument", "Company_Common_Name"]].copy()

# Define output columns (columns to retain in final file)
output_cols = [
    "Instrument", "Company_Common_Name"
]

# Function to analyze financial data with OpenAI. Its essentially a prompt that explains the objective to the AI LLM model, making sure that its predetermined to give advice with backstory so it better understand its objective
# Essentially its a guide that we write so the model is aware of the task at hand, making it better suited to make a preditction on the stock based on the information we feed it. We also allowed it to use its neural network
# To scatter the internett to find potential external factors that might help it make an informed decison. But we made sure to prompt it to NOT use any information beyond 31.12.2019, so its unbiased.
def analyze_stock_performance(row):
    prompt = f"""
   As a highly skilled financial analyst with expertise in equity research and stock performance evaluation, you are tasked with predicting the stock performance of a company in 2019 based on its financial data from 2016, 2017 and 2018. You are not allowed to analyse anything beyond the end of 2018. So the test is unbiased
   **You are allowed to research external financial news, market sentiment, industry trends, and macroeconomic conditions for 2016-2018 to improve your prediction.** However, you are NOT allowed to use any data or knowledge beyond December 31, 2018.
    Analyze the company's profitability, financial stability, growth potential, and macroeconomic risk exposure. Identify trends in revenue, net income, and financial ratios to assess how these factors might impact future stock performance. Additionally, consider external market risks, valuation metrics, and historical growth rates.

    Given the financial data for 2016, 2017, and 2018:

### **2018 Financial Data**
- Instrument: {row['Instrument']}
- Company Name: {row['Company_Common_Name']}
- Exchange Name 2018: {row['Exchange_Name_2018']}
- Revenue 2018: {row['Total_Revenue_2018']}
- Net Income 2018: {row['Net_Income_Incl_Extra_Before_Distributions_2018']}
- EBITDA 2018: {row['EBITDA_2018']}
- Gross Profit 2018: {row['Gross_Profit_2018']}
- Operating Income 2018: {row['Operating_Income_2018']}
- Free Cash Flow 2018: {row['Free_Cash_Flow_2018']}
- Current Ratio 2018: {row['Current_Ratio_2018']}
- Quick Ratio 2018: {row['Quick_Ratio_2018']}
- Total Assets 2018: {row['Total_Assets_2018']}
- Current Assets 2018: {row['Current_Assets_2018']}
- Total Liabilities 2018: {row['Total_Liabilities_2018']}
- Current Liabilities 2018: {row['Current_Liabilities_2018']}
- Enterprise Value/EBITDA 2018: {row['Enterprise_Value_To_EBITDA_(Daily_Time_Series_Ratio)_2018']}
- Monthly Closing Prices 2018: [{row['CLOSE_2018_01']}, {row['CLOSE_2018_02']}, {row['CLOSE_2018_03']}, {row['CLOSE_2018_04']}, {row['CLOSE_2018_05']}, {row['CLOSE_2018_06']}, {row['CLOSE_2018_07']}, {row['CLOSE_2018_08']}, {row['CLOSE_2018_09']}, {row['CLOSE_2018_10']}, {row['CLOSE_2018_11']}, {row['CLOSE_2018_12']}]
- Monthly Trading Volumes 2018: [{row['VOLUME_2018_01']}, {row['VOLUME_2018_02']}, {row['VOLUME_2018_03']}, {row['VOLUME_2018_04']}, {row['VOLUME_2018_05']}, {row['VOLUME_2018_06']}, {row['VOLUME_2018_07']}, {row['VOLUME_2018_08']}, {row['VOLUME_2018_09']}, {row['VOLUME_2018_10']}, {row['VOLUME_2018_11']}, {row['VOLUME_2018_12']}]

### **2017 Financial Data**
- Revenue 2017: {row['Total_Revenue_2017']}
- Net Income 2017: {row['Net_Income_Incl_Extra_Before_Distributions_2017']}
- EBITDA 2017: {row['EBITDA_2017']}
- Gross Profit 2017: {row['Gross_Profit_2017']}
- Operating Income 2017: {row['Operating_Income_2017']}
- Free Cash Flow 2017: {row['Free_Cash_Flow_2017']}
- Current Ratio 2017: {row['Current_Ratio_2017']}
- Quick Ratio 2017: {row['Quick_Ratio_2017']}
- Total Assets 2017: {row['Total_Assets_2017']}
- Current Assets 2017: {row['Current_Assets_2017']}
- Total Liabilities 2017: {row['Total_Liabilities_2017']}
- Current Liabilities 2017: {row['Current_Liabilities_2017']}
- Enterprise Value/EBITDA 2017: {row['Enterprise_Value_To_EBITDA_(Daily_Time_Series_Ratio)_2017']}
- Monthly Closing Prices 2017: [{row['CLOSE_2017_01']}, {row['CLOSE_2017_02']}, {row['CLOSE_2017_03']}, {row['CLOSE_2017_04']}, {row['CLOSE_2017_05']}, {row['CLOSE_2017_06']}, {row['CLOSE_2017_07']}, {row['CLOSE_2017_08']}, {row['CLOSE_2017_09']}, {row['CLOSE_2017_10']}, {row['CLOSE_2017_11']}, {row['CLOSE_2017_12']}]
- Monthly Trading Volumes 2017: [{row['VOLUME_2017_01']}, {row['VOLUME_2017_02']}, {row['VOLUME_2017_03']}, {row['VOLUME_2017_04']}, {row['VOLUME_2017_05']}, {row['VOLUME_2017_06']}, {row['VOLUME_2017_07']}, {row['VOLUME_2017_08']}, {row['VOLUME_2017_09']}, {row['VOLUME_2017_10']}, {row['VOLUME_2017_11']}, {row['VOLUME_2017_12']}]

### **2016 Financial Data**
- Revenue 2016: {row['Total_Revenue_2016']}
- Net Income 2016: {row['Net_Income_Incl_Extra_Before_Distributions_2016']}
- EBITDA 2016: {row['EBITDA_2016']}
- Gross Profit 2016: {row['Gross_Profit_2016']}
- Operating Income 2016: {row['Operating_Income_2016']}
- Free Cash Flow 2016: {row['Free_Cash_Flow_2016']}
- Current Ratio 2016: {row['Current_Ratio_2016']}
- Quick Ratio 2016: {row['Quick_Ratio_2016']}
- Total Assets 2016: {row['Total_Assets_2016']}
- Current Assets 2016: {row['Current_Assets_2016']}
- Total Liabilities 2016: {row['Total_Liabilities_2016']}
- Current Liabilities 2016: {row['Current_Liabilities_2016']}
- Enterprise Value/EBITDA 2016: {row['Enterprise_Value_To_EBITDA_(Daily_Time_Series_Ratio)_2016']}
- Monthly Closing Prices 2016: [{row['CLOSE_2016_01']}, {row['CLOSE_2016_02']}, {row['CLOSE_2016_03']}, {row['CLOSE_2016_04']}, {row['CLOSE_2016_05']}, {row['CLOSE_2016_06']}, {row['CLOSE_2016_07']}, {row['CLOSE_2016_08']}, {row['CLOSE_2016_09']}, {row['CLOSE_2016_10']}, {row['CLOSE_2016_11']}, {row['CLOSE_2016_12']}]
- Monthly Trading Volumes 2016: [{row['VOLUME_2016_01']}, {row['VOLUME_2016_02']}, {row['VOLUME_2016_03']}, {row['VOLUME_2016_04']}, {row['VOLUME_2016_05']}, {row['VOLUME_2016_06']}, {row['VOLUME_2016_07']}, {row['VOLUME_2016_08']}, {row['VOLUME_2016_09']}, {row['VOLUME_2016_10']}, {row['VOLUME_2016_11']}, {row['VOLUME_2016_12']}]

    
    ### **Task:**
    Based on this data, predict the company's expected stock performance in 2019. Assign a stock performance score from 1 to 10, where:
Score 1: Indicates severe financial distress, extremely high bankruptcy risk, and major instability. The firm shows negative trends across most financial indicators and has very limited ability to recover without major restructuring or external support.

Score 2: Reflects deep financial weakness and sustained poor performance. While slightly more stable than score 1, the company still faces serious solvency or liquidity issues.

Score 3: Represents high risk and weak growth prospects. Some areas may be stable, but overall performance is poor and investor confidence is typically low.

Score 4: Denotes moderate-to-high risk. Financials may be inconsistent, but the company has some structural strengths or strategic assets that could support recovery or modest growth.

Score 5: Indicates a company with stable financials and low volatility. Growth is limited, but the risk of financial collapse is low. Return potential is modest and depends on internal improvements.

Score 6: Slightly above average performance. Financially sound but lacks innovation or market momentum. A low-risk, low-growth profile.

Score 7: A solid performer with strong financial fundamentals and steady growth. Operationally efficient with a clear competitive position in its market.

Score 8: A company with robust financials, a clear strategic direction, and above-average growth. Attractive to investors looking for a balanced risk-return profile.

Score 9: Assigned to leading firms in their sector. They show outstanding financial health, consistent profitability, innovation, and market leadership. Very low risk and high return potential.

Score 10: Reserved for top-tier companies with exceptional performance across all metrics. These firms set industry standards, have durable competitive advantages, and are considered prime investment opportunities.



    **Only return a single number and nothing else.**
    """
# This code imports the ChatGPT (LLM) to the script, assigning its role in the script. The model utilized is ChatGPT 4o. 
# Temprature is how creative the model is. 0 bein that it strictly uses the data we feed it. 1 means it uses alot of external factors and makes its own dependant variables when predicting. 
# We choose a middle point with 0.5, since we want it to predict based on the data we feed it, but we also want it to make its own decisions when it comes to exteranl factors it utilizes when predicting.  
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        temperature=0.5,
        messages=[
            {"role": "system", "content": "You are an advanced financial analyst with deep expertise in stock market evaluation, capable of providing comprehensive assessments based on financial metrics and market risk indicators."},
            {"role": "user", "content": prompt}
        ]
    )

    score_text = response["choices"][0]["message"]["content"].strip()
    
    # Extract and validate score
    try:
        score = int(score_text)
    except ValueError:
        score = -1  # Assign -1 for failed cases

    return score  # Only returning score, no analysis

# Apply AI analysis to all companies (ensuring correct shape)
df['Stock_Performance_Score'] = df.apply(lambda row: analyze_stock_performance(row), axis=1)

# Restore original stock prices (overwrite scaled versions)
df.update(original_prices)

# Keep only required output columns
final_output = df[output_cols + ["Stock_Performance_Score"]]

# Display results
print(final_output[['Company_Common_Name', 'Stock_Performance_Score']])

# Save only selected columns to CSV
output_file_path = r"YOUR_DATA_HERE"
final_output.to_csv(output_file_path, index=False, float_format="%.6f")
print(f"Scores saved to: {output_file_path}")

df = pd.read_csv(r"YOUR_DATA_HERE")
df.to_excel(r"YOUR_DATA_HERE", index=False) ""
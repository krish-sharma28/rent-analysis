# U.S. Rent Trends Dashboard
 
A data science project I built to analyze how rent prices have changed across major U.S. cities since 2015, with a focus on the post-pandemic surge.
 
**Live demo:** https://rent-analysis.streamlit.app/
 
## What it does
 
- Shows rent trends over time for any combination of U.S. metros
- Ranks cities by how much rent increased since January 2020
- Displays a map of rent surges by state
- Forecasts future rent prices using Facebook Prophet (a time series forecasting model)
 
 
## Data
 
All rent data comes from the [Zillow Observed Rent Index (ZORI)](https://www.zillow.com/research/data/), which tracks average rent prices across U.S. metro areas over time.
 
## Forecasting
 
The forecast section uses [Facebook Prophet](https://facebook.github.io/prophet/), an open source forecasting library. I used it to predict what rent will look like over the next 6-24 months. It also shows a confidence range so you can see how uncertain the prediction is.
 
## Tech stack
 
- Python
- Pandas — data cleaning and analysis
- Plotly — interactive charts
- Streamlit — dashboard framework
- Prophet — rent forecasting
 

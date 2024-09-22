import streamlit as st

# Title of the landing page
st.title("Quantitative Risk Management Platform")

# Introduction to the project theme
st.markdown("""
### Welcome to the Quantitative Risk Management Platform!

In this platform, we focus on managing risk and optimizing performance for a multi-asset global portfolio using advanced quantitative finance techniques. Each project showcases a different sector of quantitative finance, yet they are all interrelated and contribute to the central goal of portfolio risk management.

Below is a breakdown of each project and how it connects to the overall theme:
""")

# Overview of Projects and their connections
st.markdown("""
### Project 1: Predictive Modeling of Asset Returns with Machine Learning
In this project, we use advanced machine learning models to predict asset returns across global markets, including stocks, bonds, and commodities. The predictions from this model feed into the portfolio optimization and stress testing stages.

---

### Project 2: Portfolio Optimization with Machine Learning and Factor Models
This project builds on the predicted returns from Project 1 and applies advanced optimization techniques to create an optimal portfolio. We use mean-variance optimization, factor models, and risk metrics like VaR and CVaR to allocate portfolio weights.

---

### Project 3: Stress Testing and Scenario Analysis
We take the optimized portfolio from Project 2 and run stress tests under extreme market conditions, simulating potential risks and tail events. This project helps evaluate how the portfolio holds up under adverse conditions.

---

### Project 4: Hedging Strategies using Derivatives
Based on the stress test outcomes, we design hedging strategies to mitigate risk. Using options, futures, and swaps, we implement strategies like delta hedging and volatility surface modeling to protect the portfolio from downside risk.

---

### GitHub Repository
Access the full codebase, documentation, and updates for all the projects in this series.
""")

# Adding buttons to navigate to the projects
st.markdown("#### Jump to Individual Projects:")

# Creating large buttons for each project
col1, col2, col3 = st.columns([1, 1, 1])
col4, col5 = st.columns([1, 1])

# Project 1 Button
with col1:
    st.markdown(f"""
    <a href="https://qrm-rsunilkumar-predictive-modeling.streamlit.app/" target="_blank">
    <button style="background-color:#4CAF50;color:white;padding:15px 32px;text-align:center;
    text-decoration:none;display:inline-block;font-size:16px;">Jump to Predictive Modeling & Asset Returns With ML</button>
    </a>
    """, unsafe_allow_html=True)

# Project 2 Button
with col2:
    st.markdown(f"""
    <a href="https://qrm-rsunilkumar-portfolio-optimization.streamlit.app/" target="_blank">
    <button style="background-color:#4CAF50;color:white;padding:15px 32px;text-align:center;
    text-decoration:none;display:inline-block;font-size:16px;">Jump to Portfolio Optimization With ML</button>
    </a>
    """, unsafe_allow_html=True)

# Project 3 Button
with col3:
    st.markdown(f"""
    <a href="https://qrm-rsunilkumar-stress-test.streamlit.app/" target="_blank">
    <button style="background-color:#4CAF50;color:white;padding:15px 32px;text-align:center;
    text-decoration:none;display:inline-block;font-size:16px;">Jump to Stress Testing</button>
    </a>
    """, unsafe_allow_html=True)

# Project 4 Button
with col4:
    st.markdown(f"""
    <a href="https://qrm-rsunilkumar-hedge-strat.streamlit.app/" target="_blank">
    <button style="background-color:#4CAF50;color:white;padding:15px 32px;text-align:center;
    text-decoration:none;display:inline-block;font-size:16px;">Jump to Hedging Strategies</button>
    </a>
    """, unsafe_allow_html=True)

# GitHub Button
with col5:
    st.markdown(f"""
    <a href="https://github.com/rahulsunilkumar/quant-risk-management/tree/main" target="_blank">
    <button style="background-color:#4CAF50;color:white;padding:15px 32px;text-align:center;
    text-decoration:none;display:inline-block;font-size:16px;">GitHub</button>
    </a>
    """, unsafe_allow_html=True)

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

# Add buttons to navigate to the projects (placeholder links for now)
st.markdown("""
#### Jump to Individual Projects:
""")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.button("Jump to Predictive Modeling & Asset Returns With ML", key="project1")
with col2:
    st.button("Jump to Portfolio Optimization With ML", key="project2")
with col3:
    st.button("Jump to Stress Testing", key="project3")
with col4:
    st.button("Jump to Hedging Strategies", key="project4")
with col5:
    st.button("GitHub", key="github")

# Placeholder URLs for now, they can be connected to the respective pages later
if st.session_state.get("project1"):
    st.write("[Go to Predictive Modeling & Asset Returns With ML](#)")
if st.session_state.get("project2"):
    st.write("[Go to Portfolio Optimization With ML](#)")
if st.session_state.get("project3"):
    st.write("[Go to Stress Testing](#)")
if st.session_state.get("project4"):
    st.write("[Go to Hedging Strategies](#)")
if st.session_state.get("github"):
    st.write("[Go to GitHub](#)")

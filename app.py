import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import traceback
import concurrent.futures
import threading
from functools import partial
import matplotlib.pyplot as plt
import io
import base64

# Import the full list of IDX tickers
from idx_all_tickers import IDX_ALL_TICKERS_YF

# Constants
MAX_TICKERS = 950  # Increased from 50 to handle all IDX stocks
DEFAULT_MIN_NI = 1.0  # Default minimum Net Income in trillion IDR
DEFAULT_MAX_PE = 15.0  # Default maximum P/E ratio
DEFAULT_MAX_PB = 1.5  # Default maximum P/B ratio
RSI_PERIOD = 25  # Period for RSI calculation
OVERSOLD_THRESHOLD = 30  # RSI threshold for oversold condition
OVERBOUGHT_THRESHOLD = 70  # RSI threshold for overbought condition
MAX_WORKERS = 10  # Maximum number of concurrent workers for parallel processing
BATCH_SIZE = 50  # Number of tickers to process in each batch

# --- Helper function for Wilder's RSI ---
def calculate_rsi_wilder(prices, period=RSI_PERIOD):
    """Calculate RSI using Wilder's smoothing method."""
    delta = prices.diff()
    
    # Ensure delta starts from index 1
    delta = delta[1:]
    
    # Make the positive gains and negative losses series
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate the initial average gain and loss using SMA for the first period
    avg_gain = gain.rolling(window=period, min_periods=period).mean()[:period]
    avg_loss = loss.rolling(window=period, min_periods=period).mean()[:period]
    
    # Calculate subsequent averages using Wilder's smoothing
    # Formula: WilderAvg = (PreviousAvg * (period - 1) + CurrentValue) / period
    for i in range(period, len(gain)):
        avg_gain = np.append(avg_gain, (avg_gain[-1] * (period - 1) + gain.iloc[i]) / period)
        avg_loss = np.append(avg_loss, (avg_loss[-1] * (period - 1) + loss.iloc[i]) / period)
        
    # Handle division by zero for avg_loss
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    
    rsi = 100 - (100 / (1 + rs))
    
    # Return the full RSI series
    return pd.Series(rsi, index=prices.index[period+1:])

# Cache technical data for 5 minutes (300 seconds)
@st.cache_data(ttl=300)
def get_rsi(ticker):
    """
    Calculate RSI for a given ticker using Wilder's smoothing.
    Returns: (rsi_value, signal, rsi_history) or None if data unavailable
    """
    try:
        # Get historical data - need enough for initial SMA + Wilder's
        # Fetching more data (e.g., 6 months) ensures robustness
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo", interval="1d")
        
        if hist.empty or len(hist["Close"]) < RSI_PERIOD + 1:
            return None
        
        # Calculate RSI using Wilder's method
        rsi_series = calculate_rsi_wilder(hist["Close"], period=RSI_PERIOD)
        
        if rsi_series.empty:
            return None
            
        # Get the latest RSI value
        latest_rsi = rsi_series.iloc[-1]
        
        # Determine signal based on RSI value
        if latest_rsi < OVERSOLD_THRESHOLD:
            signal = "Oversold"
        elif latest_rsi > OVERBOUGHT_THRESHOLD:
            signal = "Overbought"
        else:
            signal = "Neutral"
        
        # Return latest RSI, signal, and the last RSI_PERIOD values for the chart
        rsi_history = rsi_series.tail(RSI_PERIOD).values
        return (latest_rsi, signal, rsi_history)
    
    except Exception as e:
        # Log error for debugging
        st.session_state.setdefault("errors", {})
        st.session_state.errors[ticker] = f"RSI Calc Error: {e}"
        return None

# Cache fundamentals data for 24 hours (86400 seconds)
@st.cache_data(ttl=86400)
def get_fundamentals(ticker):
    """
    Retrieve fundamental financial data for a given ticker.
    Returns: (net_income, prev_net_income, pe_ratio, pb_ratio) or None if data unavailable
    """
    try:
        # Get ticker info
        stock = yf.Ticker(ticker)
        
        # Get financial data with timeout to prevent hanging
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        info = stock.info
        
        # Check if we have the necessary data
        if financials.empty or balance_sheet.empty:
            return None
        
        # Extract Net Income (convert to Trillion IDR)
        if "Net Income" in financials.index:
            net_income = financials.loc["Net Income"].iloc[0] / 1e12
            prev_net_income = financials.loc["Net Income"].iloc[1] / 1e12 if len(financials.columns) > 1 else 0
        else:
            return None
        
        # Extract P/E and P/B ratios
        pe_ratio = info.get("trailingPE", None)
        pb_ratio = info.get("priceToBook", None)
        
        # If P/E or P/B is missing, try to calculate them
        if pe_ratio is None or pb_ratio is None:
            market_cap = info.get("marketCap", None)
            if market_cap is None:
                return None
            
            if pe_ratio is None and net_income != 0:
                pe_ratio = market_cap / (net_income * 1e12)
            
            if pb_ratio is None and "Total Stockholder Equity" in balance_sheet.index:
                total_equity = balance_sheet.loc["Total Stockholder Equity"].iloc[0]
                if total_equity != 0:
                    pb_ratio = market_cap / total_equity
        
        # Return None if any value is still None or not a number
        if None in (net_income, prev_net_income, pe_ratio, pb_ratio) or \
           any(not isinstance(x, (int, float)) for x in (net_income, prev_net_income, pe_ratio, pb_ratio)):
            return None
        
        return (net_income, prev_net_income, pe_ratio, pb_ratio)
    
    except Exception as e:
        # Log error for debugging
        st.session_state.setdefault("errors", {})
        st.session_state.errors[ticker] = f"Fund. Calc Error: {e}"
        return None

def process_ticker_technical_first(ticker, rsi_min, rsi_max, show_oversold, show_overbought, show_neutral):
    """
    Process a single ticker with technical filters first.
    Returns: [ticker_symbol, rsi, signal, rsi_history] or None if not matching criteria
    """
    try:
        # Get RSI data first (now uses Wilder's method)
        rsi_data = get_rsi(ticker)
        if not rsi_data:
            return None
        
        rsi, signal, rsi_history = rsi_data
        
        # Apply RSI range filter if specified
        if (rsi_min > 0 and rsi < rsi_min) or (rsi_max < 100 and rsi > rsi_max):
            return None
        
        # Apply RSI signal filters
        if (signal == "Oversold" and not show_oversold) or \
           (signal == "Overbought" and not show_overbought) or \
           (signal == "Neutral" and not show_neutral):
            return None
        
        # Return result with technical data
        ticker_symbol = ticker.replace(".JK", "")
        return [ticker_symbol, rsi, signal, rsi_history]
    
    except Exception as e:
        # Log error for debugging
        st.session_state.setdefault("errors", {})
        st.session_state.errors[ticker] = str(e) + "\n" + traceback.format_exc()
        return None

def apply_fundamental_filters(technical_results, min_ni, max_pe, max_pb, min_growth):
    """
    Apply fundamental filters to stocks that passed technical screening.
    Returns: List of stocks with both technical and fundamental data
    """
    final_results = []
    
    for result in technical_results:
        ticker_symbol, rsi, signal, rsi_history = result
        ticker = f"{ticker_symbol}.JK"
        
        try:
            # Get fundamental data
            fund_data = get_fundamentals(ticker)
            if not fund_data:
                continue
            
            ni, prev_ni, pe, pb = fund_data
            
            # Calculate growth
            growth = ((ni - prev_ni) / abs(prev_ni) * 100) if prev_ni != 0 else 0
            
            # Apply fundamental filters
            if ni < min_ni or pe > max_pe or pb > max_pb or growth < min_growth:
                continue
            
            # Add to final results with both technical and fundamental data
            final_results.append([ticker_symbol, ni, growth, pe, pb, rsi, signal, rsi_history])
        
        except Exception as e:
            # Log error for debugging
            st.session_state.setdefault("errors", {})
            st.session_state.errors[ticker] = str(e) + "\n" + traceback.format_exc()
    
    return final_results

@st.cache_data(ttl=300)
def create_rsi_chart_image(rsi_values, current_rsi):
    """
    Create a matplotlib chart for RSI values and return as image
    Returns: image bytes
    """
    # Ensure rsi_values is a numpy array
    if isinstance(rsi_values, list):
        rsi_values = np.array(rsi_values)
        
    # Handle cases where rsi_values might be empty or too short
    if rsi_values is None or len(rsi_values) == 0:
        # Return a placeholder or empty image bytes
        fig, ax = plt.subplots(figsize=(3, 1.5))
        # Corrected ax.text call
        ax.text(0.5, 0.5, "No RSI Data", ha='center', va='center') 
        ax.set_xticks([])
        ax.set_yticks([])
        buf = io.BytesIO()
        # Corrected savefig call
        plt.savefig(buf, format='png') 
        plt.close(fig)
        buf.seek(0)
        return buf
        
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(3, 1.5))
    
    # Plot RSI line
    x = range(len(rsi_values))
    # Corrected plot call
    ax.plot(x, rsi_values, color='blue', linewidth=1.5) 
    
    # Add horizontal lines for overbought and oversold levels
    # Corrected axhline calls
    ax.axhline(y=OVERBOUGHT_THRESHOLD, color='green', linestyle='--', alpha=0.7, linewidth=1) 
    ax.axhline(y=OVERSOLD_THRESHOLD, color='red', linestyle='--', alpha=0.7, linewidth=1) 
    
    # Fill areas
    # Corrected fill_between calls
    ax.fill_between(x, OVERBOUGHT_THRESHOLD, 100, color='green', alpha=0.1) 
    ax.fill_between(x, 0, OVERSOLD_THRESHOLD, color='red', alpha=0.1) 
    
    # Set y-axis limits
    ax.set_ylim(0, 100)
    
    # Add x-axis ticks for every 5 days
    tick_positions = [i for i in range(0, len(rsi_values), 5)]
    if len(rsi_values) - 1 not in tick_positions:
        tick_positions.append(len(rsi_values) - 1)  # Add the last day
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"D-{len(rsi_values)-i}" for i in tick_positions], fontsize=7, rotation=45)
    
    # Set y-axis ticks
    ax.set_yticks([0, OVERSOLD_THRESHOLD, OVERBOUGHT_THRESHOLD, 100])
    # Corrected set_yticklabels call
    ax.set_yticklabels(['0', str(OVERSOLD_THRESHOLD), str(OVERBOUGHT_THRESHOLD), '100'], fontsize=8) 
    
    # Add current RSI value as text
    # Corrected ax.text call
    ax.text(len(rsi_values)-1, current_rsi, f' {current_rsi:.1f}', 
            verticalalignment='center', fontsize=9, 
            color='black', fontweight='bold') 
    
    # Highlight the current RSI with a dot
    # Corrected scatter call
    ax.scatter(len(rsi_values)-1, current_rsi, color='blue', s=30, zorder=5) 
    
    # Add title showing RSI period
    ax.set_title(f"RSI({RSI_PERIOD}) Chart", fontsize=10)
    
    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Tight layout
    plt.tight_layout()
    
    # Convert plot to PNG image
    buf = io.BytesIO()
    # Corrected savefig call
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.1) 
    plt.close(fig)
    buf.seek(0)
    
    return buf

def process_batch_technical_first(batch_tickers, rsi_min, rsi_max, show_oversold, show_overbought, show_neutral):
    """
    Process a batch of tickers with technical filters first.
    """
    results = []
    
    # Create a partial function with filter parameters
    process_func = partial(
        process_ticker_technical_first,
        rsi_min=rsi_min,
        rsi_max=rsi_max,
        show_oversold=show_oversold,
        show_overbought=show_overbought,
        show_neutral=show_neutral
    )
    
    # Process tickers in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        batch_results = list(executor.map(process_func, batch_tickers))
    
    # Filter out None results
    return [r for r in batch_results if r is not None]

def main():
    st.set_page_config(
        page_title="IDX Stock Screener",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better mobile responsiveness
    st.markdown("""
    <style>
    .stApp {
        max-width: 100%;
    }
    @media (max-width: 768px) {
        .stSidebar {
            width: 100%;
        }
        .stDataFrame {
            width: 100%;
            overflow-x: auto;
        }
    }
    .stProgress > div > div {
        height: 10px;
    }
    .stButton > button {
        width: 100%;
    }
    .rsi-chart-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 5px 0;
    }
    .rsi-chart-container img {
        max-width: 100%;
        height: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App header with stats
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("IDX Stock Screener")
        st.markdown(f"Screening **{len(IDX_ALL_TICKERS_YF)}** Indonesian stocks based on technical analysis first, then fundamental criteria")
    with col2:
        st.metric("Total IDX Stocks", f"{len(IDX_ALL_TICKERS_YF)}")
    
    # Initialize session state for errors and results cache
    if "errors" not in st.session_state:
        st.session_state.errors = {}
    
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = None
    
    if "results_cache" not in st.session_state:
        st.session_state.results_cache = None
    
    if "filter_settings" not in st.session_state:
        st.session_state.filter_settings = {
            "rsi_min": 0,
            "rsi_max": 100,
            "show_oversold": True,
            "show_overbought": True,
            "show_neutral": True,
            "min_ni": DEFAULT_MIN_NI,
            "max_pe": DEFAULT_MAX_PE,
            "max_pb": DEFAULT_MAX_PB,
            "min_growth": 0.0
        }
    
    # Sidebar filters with tabs for better organization
    with st.sidebar:
        st.header("IDX Stock Screener Filters")
        
        # Create tabs for filter categories
        tab1, tab2, tab3, tab4 = st.tabs(["Technical", "Fundamental", "Performance", "Settings"])
        
        with tab1:
            st.subheader("Technical Filters (First Pass)")
            st.write(f"RSI Period: {RSI_PERIOD}")
            
            # RSI range sliders
            rsi_min, rsi_max = st.slider(
                "RSI Range", 
                0, 100, 
                (0, 100),
                help="Filter stocks by RSI range"
            )
            
            # RSI signal checkboxes
            show_oversold = st.checkbox(
                "Show Oversold Stocks (RSI < 30)", 
                st.session_state.filter_settings["show_oversold"],
                help="Include stocks with RSI below 30 (potentially undervalued)"
            )
            show_overbought = st.checkbox(
                "Show Overbought Stocks (RSI > 70)", 
                st.session_state.filter_settings["show_overbought"],
                help="Include stocks with RSI above 70 (potentially overvalued)"
            )
            show_neutral = st.checkbox(
                "Show Neutral Stocks", 
                st.session_state.filter_settings["show_neutral"],
                help="Include stocks with RSI between 30 and 70"
            )
        
        with tab2:
            st.subheader("Fundamental Filters (Second Pass)")
            min_ni = st.slider(
                "Minimum Net Income (T IDR)", 
                0.1, 10.0, 
                st.session_state.filter_settings["min_ni"], 
                0.1,
                help="Minimum Net Income in trillion IDR"
            )
            max_pe = st.slider(
                "Maximum P/E Ratio", 
                5.0, 50.0, 
                st.session_state.filter_settings["max_pe"], 
                0.5,
                help="Maximum Price-to-Earnings ratio"
            )
            max_pb = st.slider(
                "Maximum P/B Ratio", 
                0.5, 5.0, 
                st.session_state.filter_settings["max_pb"], 
                0.1,
                help="Maximum Price-to-Book ratio"
            )
            min_growth = st.slider(
                "Minimum YoY Growth (%)", 
                -50.0, 100.0, 
                st.session_state.filter_settings["min_growth"], 
                5.0,
                help="Minimum Year-over-Year growth percentage"
            )
        
        with tab3:
            st.subheader("Performance Settings")
            batch_size = st.slider(
                "Batch Size", 
                10, 100, BATCH_SIZE, 10,
                help="Number of stocks to process in each batch"
            )
            max_workers = st.slider(
                "Max Concurrent Workers", 
                1, 20, MAX_WORKERS, 1,
                help="Maximum number of parallel processing threads"
            )
        
        with tab4:
            st.subheader("Refresh Settings")
            refresh = st.toggle(
                "Auto-refresh", 
                True,
                help="Automatically refresh data at specified intervals"
            )
            refresh_interval = st.slider(
                "Refresh Interval (minutes)", 
                1, 60, 5, 
                help="Time between automatic refreshes"
            ) if refresh else 0
            
            # Debug options
            st.subheader("Advanced Options")
            show_errors = st.checkbox("Show Error Log", False)
    
    # Create tabs for main content
    main_tab1, main_tab2 = st.tabs(["Screener Results", "About"])
    
    # Create progress indicators
    with main_tab1:
        progress_col1, progress_col2 = st.columns([3, 1])
        with progress_col1:
            progress_bar = st.progress(0)
        with progress_col2:
            status_text = st.empty()
        
        # Create placeholders for intermediate and final results
        technical_results_placeholder = st.empty()
        results_placeholder = st.empty()
    
    with main_tab2:
        st.subheader("About IDX Stock Screener")
        st.write(f"""
        This application screens all stocks listed on the Indonesia Stock Exchange (IDX) using a two-pass approach:
        
        **First Pass - Technical Screening:**
        - RSI({RSI_PERIOD}) with signals for oversold (RSI<{OVERSOLD_THRESHOLD}) and overbought (RSI>{OVERBOUGHT_THRESHOLD}) conditions (using Wilder's Smoothing)
        - Custom RSI range filtering
        
        **Second Pass - Fundamental Screening:**
        - Net Income > specified threshold (in trillion IDR)
        - Positive YoY growth (or as specified)
        - P/E ratio < specified threshold
        - P/B ratio < specified threshold
        
        **Data Sources:**
        - Financial data from Yahoo Finance API
        - Data is cached to minimize API calls (fundamentals: 24h, RSI: 5min)
        """)
        
        # Display current filter summary
        st.subheader("Current Filter Settings")
        filter_df = pd.DataFrame({
            "Filter": ["RSI Min", "RSI Max", "Show Oversold", "Show Overbought", "Show Neutral",
                      "Min Net Income (T IDR)", "Max P/E Ratio", "Max P/B Ratio", "Min Growth (%)"],
            "Value": [rsi_min, rsi_max, show_oversold, show_overbought, show_neutral,
                     min_ni, max_pe, max_pb, min_growth]
        })
        st.dataframe(filter_df, use_container_width=True)
        
        # Display performance metrics if available
        if st.session_state.results_cache:
            st.subheader("Performance Metrics")
            perf_df = pd.DataFrame({
                "Metric": ["Total Stocks Screened", "Technical Pass", "Final Results", "Processing Time (s)", "Errors"],
                "Value": [
                    len(IDX_ALL_TICKERS_YF),
                    st.session_state.results_cache.get("technical_count", 0),
                    st.session_state.results_cache.get("final_count", 0),
                    f"{st.session_state.results_cache.get('elapsed_time', 0):.2f}",
                    st.session_state.results_cache.get("errors", 0)
                ]
            })
            st.dataframe(perf_df, use_container_width=True)
    
    # Check if we need to refresh
    current_time = datetime.now()
    need_refresh = (
        st.session_state.last_refresh is None or
        st.session_state.results_cache is None or
        (refresh and refresh_interval > 0 and
         st.session_state.last_refresh is not None and
         (current_time - st.session_state.last_refresh).total_seconds() > refresh_interval * 60)
    )
    
    # Function to perform screening
    def perform_screening():
        technical_results = []
        final_results = []
        errors = 0
        start_time = time.time()
        
        # Clear previous errors
        st.session_state.errors = {}
        
        # FIRST PASS: Technical Screening
        # Process tickers in batches to avoid timeouts
        num_batches = (len(IDX_ALL_TICKERS_YF) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, len(IDX_ALL_TICKERS_YF))
            batch_tickers = IDX_ALL_TICKERS_YF[batch_start:batch_end]
            
            # Update progress
            progress_bar.progress((batch_idx) / (num_batches * 2))  # First half of progress bar for technical screening
            status_text.text(f"Technical Screening: Batch {batch_idx + 1}/{num_batches} (tickers {batch_start + 1}-{batch_end})")
            
            # Process batch with technical filters first
            batch_results = process_batch_technical_first(
                batch_tickers,
                rsi_min,
                rsi_max,
                show_oversold,
                show_overbought,
                show_neutral
            )
            
            # Add batch results to technical results
            technical_results.extend(batch_results)
            
            # Count errors
            errors = len(st.session_state.errors)
        
        # Update progress after technical screening
        progress_bar.progress(0.5)  # 50% complete after technical screening
        status_text.text(f"Technical Screening Complete: Found {len(technical_results)} stocks")
        
        # Remove duplicate tickers from technical results
        seen_tickers = set()
        unique_technical_results = []
        for result in technical_results:
            ticker_symbol = result[0]
            if ticker_symbol not in seen_tickers:
                seen_tickers.add(ticker_symbol)
                unique_technical_results.append(result)
        
        technical_results = unique_technical_results
        
        # Display intermediate technical results
        if technical_results:
            # Create dataframe with ticker, RSI, and signal
            tech_df = pd.DataFrame([(t[0], t[1], t[2]) for t in technical_results], 
                                  columns=["Ticker", "RSI", "Signal"])
            
            # Display the dataframe
            with technical_results_placeholder.container():
                st.subheader("Technical Screening Results")
                st.write(f"Found {len(technical_results)} stocks matching technical criteria")
                
                # Display the dataframe
                st.dataframe(tech_df, height=200, use_container_width=True)
                
                # Display RSI charts separately
                st.subheader(f"RSI({RSI_PERIOD}) Charts for Technical Results")
                
                # Create columns for displaying charts
                cols = st.columns(4)  # Display 4 charts per row
                
                # Display RSI charts
                for i, result in enumerate(technical_results):
                    ticker, rsi, signal, rsi_history = result
                    col_idx = i % 4
                    
                    with cols[col_idx]:
                        # Create chart image
                        chart_img = create_rsi_chart_image(rsi_history, rsi)
                        
                        # Display ticker and chart
                        st.write(f"**{ticker}** (RSI: {rsi:.1f})")
                        st.image(chart_img, caption=f"{signal}", use_container_width=True)
                
                st.write("Applying fundamental filters...")
        
        # SECOND PASS: Fundamental Screening
        status_text.text(f"Fundamental Screening: Processing {len(technical_results)} stocks")
        
        # Apply fundamental filters to stocks that passed technical screening
        final_results = apply_fundamental_filters(
            technical_results,
            min_ni,
            max_pe,
            max_pb,
            min_growth
        )
        
        # Remove duplicate tickers from final results
        seen_tickers = set()
        unique_final_results = []
        for result in final_results:
            ticker_symbol = result[0]
            if ticker_symbol not in seen_tickers:
                seen_tickers.add(ticker_symbol)
                unique_final_results.append(result)
        
        final_results = unique_final_results
        
        # Complete the progress bar
        progress_bar.progress(1.0)
        status_text.text(f"Screening Complete: Found {len(final_results)} stocks matching all criteria")
        
        # Update last refresh time
        st.session_state.last_refresh = current_time
        
        # Create DataFrame from results
        if final_results:
            # Create dataframe with all data
            df = pd.DataFrame([(t[0], t[1], t[2], t[3], t[4], t[5], t[6]) for t in final_results], 
                             columns=["Ticker", "NI(T)", "Growth(%)", "P/E", "P/B", "RSI", "Signal"])
            
            # Sort by Net Income (descending)
            df = df.sort_values(by="NI(T)", ascending=False)
            
            # Create a copy for display
            display_df = df.copy()
            
            # Format numeric columns
            display_df["NI(T)"] = display_df["NI(T)"].map("{:.2f}".format)
            display_df["Growth(%)"] = display_df["Growth(%)"].map("{:.2f}".format)
            display_df["P/E"] = display_df["P/E"].map("{:.2f}".format)
            display_df["P/B"] = display_df["P/B"].map("{:.2f}".format)
            
            # Cache the results
            st.session_state.results_cache = {
                "df": df,
                "display_df": display_df,
                "final_results": final_results,  # Store full results including RSI history
                "technical_count": len(technical_results),
                "final_count": len(final_results),
                "elapsed_time": time.time() - start_time,
                "errors": errors
            }
            
            return st.session_state.results_cache
        else:
            st.session_state.results_cache = {
                "df": None,
                "display_df": None,
                "final_results": [],
                "technical_count": len(technical_results),
                "final_count": 0,
                "elapsed_time": time.time() - start_time,
                "errors": errors
            }
            
            return st.session_state.results_cache
    
    # Perform screening if needed
    if need_refresh:
        results = perform_screening()
    else:
        results = st.session_state.results_cache
        # Show cached status
        status_text.text(f"Using cached results from {st.session_state.last_refresh.strftime('%H:%M:%S')}")
        progress_bar.progress(1.0)
    
    # Display results in the first tab
    with main_tab1:
        if results.get("final_count", 0) > 0:
            # Create result header with metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Technical Pass", results.get("technical_count", 0))
            with col2:
                st.metric("Final Results", results.get("final_count", 0))
            with col3:
                st.metric("Processing Time", f"{results.get('elapsed_time', 0):.2f}s")
            
            # Display results table
            with results_placeholder.container():
                st.subheader("Final Results (Technical + Fundamental)")
                
                # Display the dataframe
                st.dataframe(results["display_df"], height=300, use_container_width=True)
                
                # Display RSI charts separately
                st.subheader(f"RSI({RSI_PERIOD}) Charts for Final Results")
                
                # Create columns for displaying charts
                cols = st.columns(4)  # Display 4 charts per row
                
                # Display RSI charts for final results
                for i, result in enumerate(results.get("final_results", [])):
                    ticker, ni, growth, pe, pb, rsi, signal, rsi_history = result
                    col_idx = i % 4
                    
                    with cols[col_idx]:
                        # Create chart image
                        chart_img = create_rsi_chart_image(rsi_history, rsi)
                        
                        # Display ticker and chart
                        st.write(f"**{ticker}** (RSI: {rsi:.1f})")
                        st.image(chart_img, caption=f"{signal} | P/E: {pe:.1f} | P/B: {pb:.1f}", use_container_width=True)
                
                # Add download button for CSV export
                if results["df"] is not None:
                    csv = results["df"].to_csv(index=False)
                    st.download_button(
                        label="📥 Download results as CSV",
                        data=csv,
                        file_name=f"idx_screener_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )
        else:
            results_placeholder.info("No stocks found matching all criteria. Try adjusting your filters.")
    
    # Show error log if requested
    if show_errors and st.session_state.errors:
        with st.expander("Error Log"):
            st.write(f"Total errors: {len(st.session_state.errors)}")
            
            # Show first 10 errors to avoid cluttering the UI
            for i, (ticker, error) in enumerate(list(st.session_state.errors.items())[:10]):
                st.error(f"{ticker}: {error}")
            
            if len(st.session_state.errors) > 10:
                st.write(f"... and {len(st.session_state.errors) - 10} more errors")
    
    # Set up auto-refresh
    if refresh and refresh_interval > 0:
        next_refresh = st.session_state.last_refresh + pd.Timedelta(minutes=refresh_interval)
        st.write(f"Auto-refreshing every {refresh_interval} minutes. Next update: {next_refresh.strftime('%H:%M:%S')}")
        
        # Check if it's time to refresh
        if datetime.now() >= next_refresh:
            time.sleep(1)  # Small delay
            st.experimental_rerun()
    
    # Footer
    st.markdown(f"""
    <div style="text-align: center; margin-top: 30px; padding: 10px; border-top: 1px solid #ddd;">
        <p style="color: #666; font-size: 0.8em;">
            IDX Stock Screener | RSI({RSI_PERIOD}) Analysis (Wilder's Smoothing) | Data from Yahoo Finance | Updated: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S') if st.session_state.last_refresh else "Never"}
        </p>
    </div>
    """, 
    unsafe_allow_html=True)

if __name__ == "__main__":
    main()


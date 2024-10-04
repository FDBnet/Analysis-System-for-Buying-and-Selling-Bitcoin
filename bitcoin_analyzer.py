# CODE BY: RODRIGO S MAGALHÃES (https://github.com/FDBnet)

# WARNINGS:
""" There are two main BUGS:
- The evidence of the Interest Rate through the 'Bank of Japan' and 'Bank of England' is not working as expected;
- Obtaining the Hash Rate from the Bitcoin network is also not working """

# REPLACE "YOUR_API_KEY_HERE" WITH A REAL API! GET ONE AT "https://fredaccount.stlouisfed.org/apikey"

# CONTRIBUTE TO THE PROJECT BY DONATING BITCOIN OR SATOSHIS TO: bc1q63mezfs72jss00xvqhhjzhld33jzm322wn95x3

# US ENGLISH VERSION

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import time
import random
import json
import os
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from textblob import TextBlob
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf

# Custom logging configuration
class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            return f"\n{record.msg}"
        elif record.levelno == logging.WARNING:
            return f"\nWarning: {record.msg}"
        elif record.levelno == logging.ERROR:
            return f"\nError: {record.msg}"
        return super().format(record)

# Logger configuration
logger = logging.getLogger("BitcoinAnalyzer")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

class EnhancedBitcoinAnalyzer:
    def __init__(self):
        self.symbol = "bitcoin"
        self.vs_currency = "usd"
        self.coingecko_base_url = "https://api.coingecko.com/api/v3"
        self.cache_file = "bitcoin_data_cache.json"
        self.cache_expiry = 300  # 5 minutes in seconds
        self.session = self.create_session()
        self.google_trends_analyzer = GoogleTrendsAnalyzer()
        self.cache = self.load_cache()
        self.setup_logging()

    def create_session(self):
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)

    def setup_logging(self):
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, "bitcoin_analyzer.log")
        
        self.logger = logging.getLogger("BitcoinAnalyzer")
        self.logger.setLevel(logging.INFO)
        
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)

    def rate_limited_request(self, url, params=None, max_retries=5):
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json() if 'json' in response.headers.get('Content-Type', '').lower() else response
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Failed to get data after {max_retries} attempts.")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        return None
    
    def mempool_api_request(self, endpoint):
        url = f"https://mempool.space/api/{endpoint}"
        try:
            response = self.rate_limited_request(url)
            if response:
                return response
        except Exception as e:
            self.logger.error(f"Error making request to {url}: {str(e)}")
        return None

    def get_historical_data(self, days=200):
        url = f"{self.coingecko_base_url}/coins/{self.symbol}/market_chart"
        params = {"vs_currency": self.vs_currency, "days": days}
        data = self.rate_limited_request(url, params)
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        if 'total_volumes' in data:
            volume_df = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
            volume_df.set_index('timestamp', inplace=True)
            df = df.join(volume_df)
        
        return df

    def get_current_price(self):
        url = f"{self.coingecko_base_url}/simple/price"
        params = {"ids": self.symbol, "vs_currencies": self.vs_currency}
        data = self.rate_limited_request(url, params)
        if data and self.symbol in data:
            return data[self.symbol][self.vs_currency]
        return None

    def calculate_200_day_ma(self):
        df = self.get_historical_data()
        if df.empty:
            return None
        return df['price'].rolling(window=200).mean().iloc[-1]

    def compare_price_to_ma(self):
        current_price = self.get_current_price()
        ma_200 = self.calculate_200_day_ma()
        
        if current_price is None or ma_200 is None:
            return "Unable to compare due to insufficient data."
        
        if current_price > ma_200:
            return f"|Current price (${current_price:.2f}) is greater than the 200-day moving average (${ma_200:.2f})."
        else:
            return f"|The current price (${current_price:.2f}) is less than or equal to the 200-day moving average (${ma_200:.2f})."

    def estimate_funding_rate(self):
        spot_price = self.get_current_price()
        if spot_price is None:
            return None

        recent_data = self.get_historical_data(days=1)
        if recent_data.empty:
            return None

        if 'volume' in recent_data.columns:
            volume_weighted_price = (recent_data['price'] * recent_data['volume']).sum() / recent_data['volume'].sum()
        else:
            volume_weighted_price = recent_data['price'].mean()

        price_difference = (volume_weighted_price - spot_price) / spot_price
        estimated_funding_rate = price_difference * 3 * 100  # Converted to percentage
        return estimated_funding_rate

    def get_exchange_funding_rate(self, exchange, url, params, data_path):
        data = self.rate_limited_request(url, params)
        if data:
            for key in data_path:
                if isinstance(data, list) and isinstance(key, int) and key < len(data):
                    data = data[key]
                elif isinstance(data, dict) and key in data:
                    data = data[key]
                else:
                    logger.warning(f"Failed to get Funding Rate from {exchange}.")
                    return None
            try:
                return float(data) * 100  # Converting to percentage
            except (ValueError, TypeError):
                logger.warning(f"Failed to convert Funding Rate from {exchange}.")
                return None
        logger.error(f"Unable to retrieve data from {exchange}.")
        return None

    def get_binance_funding_rate(self):
        return self.get_exchange_funding_rate(
            "Binance",
            "https://fapi.binance.com/fapi/v1/fundingRate",
            {"symbol": "BTCUSDT"},
            [0, 'fundingRate']
        )

    def get_bybit_funding_rate(self):
        try:
            rate = self.get_exchange_funding_rate(
                "Bybit v5",
                "https://api.bybit.com/v5/market/tickers",
                {"category": "linear", "symbol": "BTCUSDT"},
                ['result', 'list', 0, 'fundingRate']
            )
            if rate is not None:
                return rate
            
            self.logger.warning("Failed to get funding rate from Bybit v5.")
            return None
        except Exception as e:
            self.logger.error(f"Error getting funding rate from Bybit: {str(e)}")
            return None

    def get_okex_funding_rate(self):
        return self.get_exchange_funding_rate(
            "OKX",
            "https://www.okx.com/api/v5/public/funding-rate",
            {"instId": "BTC-USD-SWAP"},
            ['data', 0, 'fundingRate']
        )

    def calculate_rsi(self, data, window=14):
        delta = data['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] # Returns the most recent RSI

    def get_rsi_evaluation(self):
        try:
            df = self.get_historical_data(days=30)  # We take 30 days of data to calculate the RSI
            if df.empty:
                logger.warning("Insufficient data to calculate RSI")
                return None, "Unable to calculate RSI due to insufficient data."
            
            rsi = self.calculate_rsi(df)
            
            if rsi > 70:
                return rsi, "The market may be overbought. Consider selling or be cautious about buying."
            elif rsi < 30:
                return rsi, "The market may be oversold. This could be a buying opportunity, but keep an eye on other indicators."
            else:
                return rsi, "RSI is in a neutral range. Consider other indicators to make decisions."
        
        except Exception as e:
            logger.error(f"RSI Error: {e}")
            return None, f"Unable to calculate RSI due to an error: {e}"

    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        try:
            # Calculates exponential moving averages
            ema_fast = data['price'].ewm(span=fast_period, adjust=False).mean()
            ema_slow = data['price'].ewm(span=slow_period, adjust=False).mean()
            
            # Calculate the MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate the signal line
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # Calculate the histogram
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.error(f"MACD Error: {e}")
            return None, None, None

    def get_macd_evaluation(self):
        try:
            df = self.get_historical_data(days=60)  # We take 60 days of data to calculate MACD
            if df.empty:
                logger.warning("Insufficient data to calculate MACD")
                return None, "Insufficient data to calculate MACD"
            
            macd_line, signal_line, histogram = self.calculate_macd(df)
            
            if macd_line is None or signal_line is None:
                return None, "Unable to calculate MACD"
            
            # We take the last values for comparison
            last_macd = macd_line.iloc[-1]
            last_signal = signal_line.iloc[-1]
            last_histogram = histogram.iloc[-1]
            prev_histogram = histogram.iloc[-2]
            
            if last_macd > last_signal:
                if last_histogram > 0 and last_histogram > prev_histogram:
                    interpretation = "The MACD is above the signal line and the histogram is rising. This could indicate strong bullish momentum."
                else:
                    interpretation = "The MACD is above the signal line. This could indicate an uptrend, but watch the histogram for confirmation."
            elif last_macd < last_signal:
                if last_histogram < 0 and last_histogram < prev_histogram:
                    interpretation = "The MACD is below the signal line and the histogram is falling. This could indicate strong bearish momentum."
                else:
                    interpretation = "The MACD is below the signal line. This could indicate a downtrend, but watch the histogram for confirmation."
            else:
                interpretation = "The MACD and the signal line are close together. The market may be at a point of indecision."
            
            return (last_macd, last_signal, last_histogram), interpretation
        
        except Exception as e:
            logger.error(f"Error evaluating MACD: {e}")
            return None, f"Unable to evaluate MACD due to error: {e}"

    def get_supply_distribution(self):
        try:
            url = f"{self.coingecko_base_url}/coins/bitcoin"
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false",
                "sparkline": "false"
            }
            data = self.rate_limited_request(url, params)
            if not data or 'market_data' not in data:
                logger.warning("Unable to retrieve supply distribution data")
                return None

            supply_distribution = {
                'circulating': data['market_data']['circulating_supply'],
                'total': data['market_data']['total_supply'],
                'max': data['market_data']['max_supply']
            }

            return supply_distribution

        except Exception as e:
            logger.error(f"Error getting supply distribution data: {e}")
            return None

    def analyze_supply_distribution(self, supply_distribution):
        if not supply_distribution:
            return "It was not possible to analyze the supply distribution due to lack of data."

        circulating = supply_distribution['circulating']
        total = supply_distribution['total']
        max_supply = supply_distribution['max']

        percent_circulating = (circulating / max_supply) * 100
        percent_non_circulating = ((total - circulating) / max_supply) * 100
        percent_non_mined = ((max_supply - total) / max_supply) * 100

        analysis = f"| {percent_circulating:.2f}% of the maximum supply is in circulation.\n"
        analysis += f"| {percent_non_circulating:.2f}% of the maximum supply has been mined but is not in circulation.\n"
        analysis += f"| {percent_non_mined:.2f}% of the maximum supply has not yet been mined.\n"

        if percent_circulating > 90:
            analysis += "High proportion of circulating supply, which may indicate wide distribution.\n"
        elif percent_non_circulating > 10:
            analysis += "Significant proportion of non-circulating supply, which may indicate accumulation.\n"

        return analysis

    def calculate_epr(self, price_data, window=7):
        """
        Calculates the Estimated Price Ratio (EPR), an approximation of the SOPR.

        :param price_data: DataFrame with historical prices
        :param window: Window for calculating the moving average (in days)
        :return: Series with the calculated EPR
        """
        # Calculate the daily return
        daily_return = price_data['price'].pct_change()
        
        # Calculate the moving average of the returns
        moving_avg_return = daily_return.rolling(window=window).mean()
        
        # Calculate the EPR
        epr = (1 + daily_return) / (1 + moving_avg_return)
        
        return epr

    def analyze_epr(self, epr_data):
        if epr_data is None or epr_data.empty:
            return "Insufficient EPR data for analysis."
        
        latest_epr = epr_data.iloc[-1]
        avg_epr = epr_data.mean()
        
        analysis = f"|Current EPR: {latest_epr:.4f}\n"
        analysis += f"|Average EPR (7 days): {avg_epr:.4f}\n"
        
        if latest_epr > 1:
            if latest_epr > avg_epr:
                analysis += "The EPR is above 1 and above the average, suggesting that recent prices are above the short-term trend. This could indicate bullish momentum in the market."
            else:
                analysis += "The EPR is above 1 but close to the average, suggesting a balance between optimism and profit-taking."
        else:
            if latest_epr < avg_epr:
                analysis += "The EPR is below 1 and below the average, suggesting that recent prices are below the short-term trend. This may indicate a bearish moment in the market."
            else:
                analysis += "The EPR is below 1 but close to the average, suggesting that the market may be approaching a turning point."
        
        return analysis
    
    def calculate_nvt_ratio(self, days=30):
        """
        Calculates an approximation of the NVT Ratio using market capitalization and trading volume.
        
        :param days: Number of days to collect data
        :return: DataFrame with the calculated NVT Ratio
        """
        url = f"{self.coingecko_base_url}/coins/{self.symbol}/market_chart"
        params = {
            "vs_currency": self.vs_currency,
            "days": days
        }
        
        data = self.rate_limited_request(url, params)
        if not data:
            logger.warning("Unable to obtain data for NVT Ratio calculation")
            return None

        df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
        market_caps['timestamp'] = pd.to_datetime(market_caps['timestamp'], unit='ms')
        market_caps.set_index('timestamp', inplace=True)
        
        total_volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
        total_volumes['timestamp'] = pd.to_datetime(total_volumes['timestamp'], unit='ms')
        total_volumes.set_index('timestamp', inplace=True)
        
        df = df.join(market_caps).join(total_volumes)
        
        # Calculate the NVT Ratio
        df['nvt_ratio'] = df['market_cap'] / df['volume']
        
        return df

    def analyze_nvt_ratio(self, nvt_data):
        if nvt_data is None or nvt_data.empty:
            return "Insufficient NVT data for analysis."
        
        latest_nvt = nvt_data['nvt_ratio'].iloc[-1]
        avg_nvt = nvt_data['nvt_ratio'].mean()
        
        analysis = f"|Current NVT Ratio: {latest_nvt:.2f}\n"
        analysis += f"|Average NVT Ratio (30 days): {avg_nvt:.2f}\n"
        
        if latest_nvt > avg_nvt * 1.5:
            analysis += "The NVT Ratio is significantly above average, suggesting that the network value may be overvalued relative to economic activity."
        elif latest_nvt > avg_nvt * 1.2:
            analysis += "The NVT Ratio is above average, indicating possible overvaluation, but still within reasonable limits."
        elif latest_nvt < avg_nvt * 0.8:
            analysis += "The NVT Ratio is below average, suggesting that the network value may be undervalued relative to economic activity."
        else:
            analysis += "The NVT Ratio is close to average, indicating a balanced valuation relative to economic activity."
        
        return analysis
    
    def get_network_hashrate(self, days=30):
        """
        Gets hash rate data for the Bitcoin network using the mempool.space API.
        
        :param days: Number of days to collect data
        :return: DataFrame with hash rate data
        """
        endpoint = f"v1/mining/hashrate/{days * 144}"  # 144 blocks per day on average
        data = self.mempool_api_request(endpoint)
        
        if not data:
            self.logger.warning("Unable to obtain network hash rate data")
            return None

        df = pd.DataFrame(data, columns=['timestamp', 'hashrate'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        df['hashrate'] = df['hashrate'] / 1e9  # Converting to EH/s
        
        return df
    
    def get_transaction_fees(self):
        """
        Gets data about current transaction fees on the Bitcoin network.
        
        :return: Dict with information about transaction fees
        """
        endpoint = "v1/fees/recommended"
        data = self.mempool_api_request(endpoint)
        
        if not data:
            self.logger.warning("Unable to obtain transaction fee data")
            return None

        return data

    def analyze_network_hashrate(self, hashrate_data):
        if hashrate_data is None or hashrate_data.empty:
            return "Insufficient hash rate data for analysis."
        
        latest_hashrate = hashrate_data['hashrate'].iloc[-1]
        avg_hashrate = hashrate_data['hashrate'].mean()
        
        if np.isnan(latest_hashrate) or np.isnan(avg_hashrate):
            return "Unable to calculate hash rate due to invalid data."
        
        pct_change = ((latest_hashrate - hashrate_data['hashrate'].iloc[0]) / hashrate_data['hashrate'].iloc[0]) * 100
        
        analysis = f"|Current Hash Rate: {latest_hashrate:.2f} EH/s\n"
        analysis += f"|Average Hash Rate (30 days): {avg_hashrate:.2f} EH/s\n"
        analysis += f"|Percentage change (30 days): {pct_change:.2f}%\n"
        
        if latest_hashrate > avg_hashrate * 1.1:
            analysis += "The current hash rate is significantly above average, indicating an increase in network security and possibly greater interest from miners."
        elif latest_hashrate < avg_hashrate * 0.9:
            analysis += "The current hash rate is below average, which may indicate a reduction in mining activity or changes in market conditions."
        else:
            analysis += "The current hash rate is close to average, suggesting stability in mining activity."
        
        if pct_change > 10:
            analysis += "\nThe significant increase in hash rate may indicate greater confidence in the network and potentially a bullish signal for the price."
        elif pct_change < -10:
            analysis += "\nThe significant decrease in hash rate may indicate challenges for miners or regulatory changes, potentially a bearish signal."
        
        return analysis
    
    def get_global_interest_rates(self):
        interest_rates = {
            '- FED (USA)': self.get_fed_rate(),
            '- ECB (Europe)': self.get_ecb_rate(),
            '- BoJ (Japan)': self.get_boj_rate(),
            '- BoE (UK)': self.get_boe_rate()
        }
        
        # Remove None values
        interest_rates = {k: v for k, v in interest_rates.items() if v is not None}
        
        if not interest_rates:
            self.logger.warning("Unable to obtain any global interest rates.")
            return None
        
        return pd.DataFrame(list(interest_rates.items()), columns=['Central Bank', 'Interest Rate'])

    def get_fed_rate(self):
        try:
            # Primary method: scraping from the Federal Reserve website
            url = "https://www.federalreserve.gov/releases/h15/"
            response = self.rate_limited_request(url)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                rate_element = soup.find('th', id='id94d1cc0', string='Federal funds (effective)')
                if rate_element:
                    rate = rate_element.find_next('td', class_='data').text
                    return float(rate.strip())
            
            # Secondary method: FRED API (Federal Reserve Economic Data)
            fred_url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": "FEDFUNDS",
                "api_key": "YOUR_API_KEY_HERE",
                "sort_order": "desc",
                "limit": 1,
                "file_type": "json"
            }
            fred_data = self.rate_limited_request(fred_url, params)
            if fred_data and 'observations' in fred_data:
                return float(fred_data['observations'][0]['value'])
            
            self.logger.error("Failed to get FED rate from all sources.")
            return None
        except Exception as e:
            self.logger.error(f"Error getting FED rate: {str(e)}")
            return None
        
    def get_ecb_rate(self):
        try:
            url = "https://www.ecb.europa.eu/stats/policy_and_exchange_rates/key_ecb_interest_rates/html/index.en.html"
            response = self.rate_limited_request(url)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                rate_table = soup.find('table')
                if rate_table:
                    rows = rate_table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 3:
                            date = cells[0].text.strip()
                            deposit_rate = cells[1].text.strip()
                            main_rate = cells[2].text.strip()
                            if date and deposit_rate and main_rate:
                                return float(main_rate)
            self.logger.error("Unable to find ECB rates on the page.")
            return None
        except Exception as e:
            self.logger.error(f"Error getting ECB rate: {str(e)}")
            return None

    def get_boj_rate(self):
        try:
            url = "https://www.boj.or.jp/en/statistics/boj/other/discount/index.htm/"
            response = self.rate_limited_request(url)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                rate_element = soup.find('td', string='Basic Discount Rate and Basic Loan Rate')
                if rate_element:
                    rate = rate_element.find_next('td').text
                    return float(rate.strip().replace('%', ''))
            return None
        except Exception as e:
            self.logger.error(f"Error getting BOJ rate: {str(e)}")
            return None

    def get_boe_rate(self):
        try:
            url = "https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp"
            response = self.rate_limited_request(url)
            if response and isinstance(response, requests.Response):
                soup = BeautifulSoup(response.text, 'html.parser')
                rate_element = soup.select_one('.featured-stat .stat-figure')
                if rate_element:
                    rate = rate_element.text.strip().replace('%', '')
                    return float(rate)
            self.logger.warning("Unable to get the Bank of England rate.")
            return None
        except Exception as e:
            self.logger.error(f"Error getting BOE rate: {str(e)}")
            return None

    def analyze_global_interest_rates(self, rates_data):
        if rates_data is None or rates_data.empty:
            return "Insufficient global interest rate data for analysis."
        
        analysis = ""
        for _, row in rates_data.iterrows():
            analysis += f"{row['Central Bank']}: {row['Interest Rate']:.2f}%\n"
        
        avg_rate = rates_data['Interest Rate'].mean()
        analysis += f"\n|Global average rate: {avg_rate:.2f}%\n"
        
        if avg_rate < 1:
            analysis += "\nGlobal interest rates are very low, which is generally favorable for risk assets like Bitcoin."
        elif avg_rate < 3:
            analysis += "\nGlobal interest rates are moderate, which may still be favorable for Bitcoin, but with less intensity."
        else:
            analysis += "\nGlobal interest rates are relatively high, which may put pressure on risk assets like Bitcoin."
        
        return analysis
    
    def analyze_transaction_fees(self, fee_data):
        if fee_data is None:
            return "Insufficient transaction fee data for analysis."
        
        analysis = ""
        analysis += f"|Fast fee (next block): {fee_data['fastestFee']} sat/vB\n"
        analysis += f"|Average fee (half hour): {fee_data['halfHourFee']} sat/vB\n"
        analysis += f"|Economic fee (1 hour): {fee_data['hourFee']} sat/vB\n"
        
        if fee_data['fastestFee'] > 100:
            analysis += "\nTransaction fees are very high, indicating high demand on the network."
        elif fee_data['fastestFee'] > 50:
            analysis += "\nTransaction fees are moderately high, suggesting significant demand."
        elif fee_data['fastestFee'] < 10:
            analysis += "\nTransaction fees are low, indicating little network congestion."
        else:
            analysis += "\nTransaction fees are at normal levels."
        
        return analysis
    
    def get_recent_blocks(self, max_retries=5, delay=5):
        """
        Gets data for the most recent blocks from the public mempool.space API
        
        :param max_retries: Maximum number of retry attempts
        :param delay: Wait time between attempts (in seconds)
        :return: List of dictionaries with block data or None in case of failure
        """
        url = "https://mempool.space/api/v1/blocks"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))
        
        self.logger.error(f"Failed to get recent block data after {max_retries} attempts")
        return None

    def analyze_recent_blocks(self, blocks_data):
        if not blocks_data:
            return "Recent block data unavailable for analysis."
        
        analysis = "Analysis of Recent Blocks:\n"
        avg_block_size = sum(block['size'] for block in blocks_data) / len(blocks_data)
        avg_tx_count = sum(block['tx_count'] for block in blocks_data) / len(blocks_data)
        
        analysis += f"|Average block size: {avg_block_size / 1000:.2f} KB\n"
        analysis += f"|Average transactions per block: {avg_tx_count:.0f}\n"
        
        if avg_block_size > 1_300_000:  # 1.3 MB
            analysis += "Blocks are nearly full, indicating high network demand.\n"
        elif avg_block_size < 500_000:  # 500 KB
            analysis += "Blocks are relatively empty, suggesting low network demand.\n"
        else:
            analysis += "Block sizes are at normal levels.\n"
        
        return analysis
    
    def get_mempool_data(self, max_retries=5, delay=5):
        """
        Gets mempool data from the public mempool.space API
        
        :param max_retries: Maximum number of retry attempts
        :param delay: Wait time between attempts (in seconds)
        :return: Dictionary with mempool data or None in case of failure
        """
        url = "https://mempool.space/api/v1/mempool"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))
        
        self.logger.error(f"Failed to get mempool data after {max_retries} attempts")
        return None

    def analyze_mempool(self, mempool_data):
        if not mempool_data:
            return "Mempool data unavailable for analysis."
        
        analysis = "Mempool Analysis:\n"
        analysis += f"|Mempool size: {mempool_data['vsize'] / 1_000_000:.2f} MB\n"
        analysis += f"|Number of transactions: {mempool_data['count']}\n"
        
        if mempool_data['vsize'] > 80_000_000:  # 80 MB
            analysis += "The mempool is very congested. Expect high transaction fees.\n"
        elif mempool_data['vsize'] < 5_000_000:  # 5 MB
            analysis += "The mempool is relatively empty. Transaction fees should be low.\n"
        else:
            analysis += "The mempool is at normal levels.\n"
        
        return analysis
    
    def get_mining_difficulty(self, max_retries=5, delay=5):
        """
        Gets mining difficulty data from the public mempool.space API
        
        :param max_retries: Maximum number of retry attempts
        :param delay: Wait time between attempts (in seconds)
        :return: Dictionary with difficulty data or None in case of failure
        """
        url = "https://mempool.space/api/v1/difficulty-adjustment"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))
        
        self.logger.error(f"Failed to get difficulty data after {max_retries} attempts")
        return None

    def analyze_mining_difficulty(self, difficulty_data):
        if not difficulty_data:
            return "Mining difficulty data unavailable for analysis."
        
        analysis = "Mining Difficulty Analysis:\n"
        
        try:
            current_difficulty = difficulty_data.get('current_difficulty')
            if current_difficulty is not None:
                analysis += f"|Current difficulty: {current_difficulty:,}\n"
            else:
                analysis += "Current difficulty not available.\n"
            
            estimated_retarget = difficulty_data.get('estimated_retarget_percentage')
            if estimated_retarget is not None:
                analysis += f"|Estimated change: {estimated_retarget:.2f}%\n"
            else:
                analysis += "Estimated change not available.\n"
            
            remaining_blocks = difficulty_data.get('remaining_blocks')
            if remaining_blocks is not None:
                analysis += f"|Blocks until adjustment: {remaining_blocks}\n"
            else:
                analysis += "Number of blocks until adjustment not available.\n"
            
            if estimated_retarget is not None:
                if estimated_retarget > 5:
                    analysis += "Difficulty is likely to increase, indicating an increase in mining capacity.\n"
                elif estimated_retarget < -5:
                    analysis += "Difficulty is likely to decrease, possibly indicating a reduction in mining capacity.\n"
                else:
                    analysis += "Difficulty is expected to remain relatively stable.\n"
        except Exception as e:
            self.logger.error(f"Error analyzing mining difficulty: {str(e)}")
            analysis += "An error occurred while analyzing mining difficulty data.\n"
        
        return analysis

    def get_bitcoin_dominance(self):
        """
        Gets the current Bitcoin dominance in the cryptocurrency market.
        
        :return: Float representing the percentage of Bitcoin dominance or None if unable to obtain the data
        """
        url = "https://api.coingecko.com/api/v3/global"
        try:
            data = self.rate_limited_request(url)
            
            if data and isinstance(data, dict) and 'data' in data:
                market_data = data['data']
                if 'market_cap_percentage' in market_data and 'btc' in market_data['market_cap_percentage']:
                    return market_data['market_cap_percentage']['btc']
            
            self.logger.warning("Unable to obtain Bitcoin dominance data from the API")
            return None
        except Exception as e:
            self.logger.error(f"Error getting Bitcoin dominance: {str(e)}")
            return None

    def analyze_bitcoin_dominance(self, dominance):
        if dominance is None:
            return "Insufficient Bitcoin dominance data for analysis."
        
        analysis = f"Bitcoin Dominance: {dominance:.2f}%\n"
        
        if dominance > 60:
            analysis += "High Bitcoin dominance suggests strong confidence in BTC relative to other cryptocurrencies."
        elif dominance < 40:
            analysis += "Low Bitcoin dominance may indicate increased interest in altcoins or a decrease in confidence in BTC."
        else:
            analysis += "Bitcoin dominance is at moderate levels, suggesting a balance between BTC and altcoins."
        
        return analysis

    def analyze(self):
        self.logger.info("\nStarting Bitcoin market analysis...")
        print("\nAnalyzing the Bitcoin market...", end="", flush=True)
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {
                executor.submit(self.compare_price_to_ma): "price_comparison",
                executor.submit(self.estimate_funding_rate): "estimated_funding_rate",
                executor.submit(self.get_binance_funding_rate): "binance_rate",
                executor.submit(self.get_bybit_funding_rate): "bybit_rate",
                executor.submit(self.get_okex_funding_rate): "okex_rate",
                executor.submit(self.google_trends_analyzer.get_google_trends_data): "sentiment_score",
                executor.submit(self.get_rsi_evaluation): "rsi_data",
                executor.submit(self.get_macd_evaluation): "macd_data",
                executor.submit(self.get_supply_distribution): "supply_distribution_data",
                executor.submit(self.get_historical_data, days=30): "price_data_for_epr",
                executor.submit(self.calculate_nvt_ratio): "nvt_data",
                executor.submit(self.get_network_hashrate): "hashrate_data",
                executor.submit(self.get_global_interest_rates): "global_rates_data",
                executor.submit(self.get_transaction_fees): "transaction_fees",
                executor.submit(self.get_recent_blocks): "recent_blocks",
                executor.submit(self.get_mining_difficulty): "mining_difficulty",
                executor.submit(self.get_bitcoin_dominance): "bitcoin_dominance",
            }
            
            results = {}

            for future in as_completed(futures):
                try:
                    results[futures[future]] = future.result()
                    print(".", end="", flush=True)
                except Exception as e:
                    self.logger.error(f"Error getting {futures[future]}: {str(e)}")
                    results[futures[future]] = None
                    print("x", end="", flush=True)

        print("\nAnalysis completed!")

        # Calculate EPR
        price_data_for_epr = results.get('price_data_for_epr')
        if price_data_for_epr is not None and not price_data_for_epr.empty:
            epr_data = self.calculate_epr(price_data_for_epr)
            results['epr_data'] = epr_data
        else:
            results['epr_data'] = None

        self.print_analysis_results(results)

    def get_master_evaluation(self, results):
        try:
            price_trend = 1 if "greater than" in results.get('price_comparison', '') else -1
            
            funding_rates = [
                results.get('estimated_funding_rate'),
                results.get('binance_rate'),
                results.get('bybit_rate'),
                results.get('okex_rate')
            ]
            valid_rates = [r for r in funding_rates if r is not None]
            avg_funding_rate = np.mean(valid_rates) if valid_rates else 0
            
            sentiment_score = results.get('sentiment_score', 50)  # Neutral value if not available
            
            rsi_data = results.get('rsi_data', (None, ''))
            rsi_value, _ = rsi_data if isinstance(rsi_data, tuple) else (None, '')
            rsi_score = 0
            if rsi_value is not None:
                if rsi_value > 70:
                    rsi_score = -1
                elif rsi_value < 30:
                    rsi_score = 1
            
            macd_data = results.get('macd_data', (None, ''))
            macd_values, _ = macd_data if isinstance(macd_data, tuple) else (None, '')
            macd_score = 0
            if macd_values is not None:
                macd, signal, _ = macd_values
                if macd > signal:
                    macd_score = 1
                elif macd < signal:
                    macd_score = -1
        
            supply_distribution_data = results['supply_distribution_data']
            supply_distribution_score = 0
            if supply_distribution_data:
                percent_circulating = (supply_distribution_data['circulating'] / supply_distribution_data['max']) * 100
                if percent_circulating > 90:
                    supply_distribution_score = 1  # Positive signal (wide distribution)
                elif percent_circulating < 80:
                    supply_distribution_score = -1  # Negative signal (possible concentration)
            
            epr_data = results['epr_data']
            epr_score = 0
            if epr_data is not None and not epr_data.empty:
                latest_epr = epr_data.iloc[-1]
                if latest_epr > 1:
                    epr_score = 1  # Possible bullish moment
                elif latest_epr < 1:
                    epr_score = -1   # Possible bearish moment
            
            nvt_data = results['nvt_data']
            nvt_score = 0
            if nvt_data is not None and not nvt_data.empty:
                latest_nvt = nvt_data['nvt_ratio'].iloc[-1]
                avg_nvt = nvt_data['nvt_ratio'].mean()
                if latest_nvt > avg_nvt * 1.2:
                    nvt_score = -1  # Possible overvaluation
                elif latest_nvt < avg_nvt * 0.8:
                    nvt_score = 1   # Possible undervaluation
            
            hashrate_data = results['hashrate_data']
            hashrate_score = 0
            if hashrate_data is not None and not hashrate_data.empty:
                latest_hashrate = hashrate_data['hashrate'].iloc[-1]
                avg_hashrate = hashrate_data['hashrate'].mean()
                pct_change = ((latest_hashrate - hashrate_data['hashrate'].iloc[0]) / hashrate_data['hashrate'].iloc[0]) * 100
                
                if latest_hashrate > avg_hashrate * 1.1 or pct_change > 10:
                    hashrate_score = 1  # Positive signal
                elif latest_hashrate < avg_hashrate * 0.9 or pct_change < -10:
                    hashrate_score = -1  # Negative signal
            
            global_rates_data = results['global_rates_data']
            global_rates_score = 0
            if global_rates_data is not None and not global_rates_data.empty:
                avg_rate = global_rates_data['Interest Rate'].mean()
                if avg_rate < 1:
                    global_rates_score = 1  # Favorable for Bitcoin
                elif avg_rate > 3:
                    global_rates_score = -1  # Less favorable for Bitcoin
            
            # Adjust weights to include Global Interest Rates
            price_weight = 0.15
            funding_weight = 0.11
            sentiment_weight = 0.11
            rsi_weight = 0.15
            macd_weight = 0.11
            supply_distribution_weight = 0.11
            epr_weight = 0.07
            nvt_weight = 0.07
            hashrate_weight = 0.06
            global_rates_weight = 0.06
            
            # Final score calculation (including Global Interest Rates)
            final_score = (
                price_trend * price_weight +
                (-1 if avg_funding_rate > 0 else 1) * funding_weight +
                (sentiment_score / 50 - 1) * sentiment_weight +
                rsi_score * rsi_weight +
                macd_score * macd_weight +
                supply_distribution_score * supply_distribution_weight +
                epr_score * epr_weight +
                nvt_score * nvt_weight +
                hashrate_score * hashrate_weight +
                global_rates_score * global_rates_weight
            )
            
            # Interpretação do score final
            if final_score > 0.5:
                return "BUY", final_score
            elif final_score < -0.5:
                return "SELL", final_score
            else:
                return "HOLD", final_score
        except Exception as e:
            self.logger.error(f"Error in final assessment: {str(e)}")
            return "INCONCLUSIVE", 0

    def print_analysis_results(self, results):
        print("\n" + "="*50)
        print(" ## BITCOIN MARKET ANALYSIS ##")
        print("="*50 + "\n")
        print(results['price_comparison'])
        print("\n # Funding Rates:")
        print(f"{'Exchange':<12} {'Rate':>12}")
        print("-"*22)
        for exchange, rate_key in [("- Estimate", "estimated_funding_rate"), 
                                ("- Binance", "binance_rate"), 
                                ("- Bybit", "bybit_rate"), 
                                ("- OKX", "okex_rate")]:
            if results[rate_key] is not None:
                print(f"{exchange:<10} {results[rate_key]:>10.4f}%")
        
        print("\nNote: These are real-time rates and may not reflect")
        print("the exact Funding Rate for the next period.")
        
        valid_rates = [r for r in [results['binance_rate'], results['bybit_rate'], results['okex_rate']] if r is not None]
        if valid_rates:
            avg_rate = np.mean(valid_rates)
            print(f"\n|Average Funding Rates: {avg_rate:.4f}%")
            if abs(avg_rate) > 0.1:  # 0.1%
                print("Attention: Average Funding Rates is high,")
                print("indicating possible volatility in the market.")
            elif abs(avg_rate) < 0.01:  # 0.01%
                print("Average Funding Rates is at low levels,")
                print("indicating possible stability in the market.")
            else:
                print("Average Funding Rates is at moderate levels.")
        else:
            print("Unable to calculate the average Funding Rates")
            print("due to lack of data.")
        
        print("\n # Google Trends Sentiment Analysis:")
        sentiment_score = results['sentiment_score']
        print(f"|Sentiment Score: {sentiment_score:.2f}")
        if sentiment_score > 70:
            print("The market sentiment is very positive.")
        elif sentiment_score > 50:
            print("The market sentiment is slightly positive.")
        elif sentiment_score > 30:
            print("The market sentiment is neutral.")
        else:
            print("The market sentiment is negative.")
        
        print("\n # RSI (Relative Strength Index) Analysis:")
        rsi_value, rsi_interpretation = results['rsi_data']
        if rsi_value is not None:
            print(f"|Current RSI: {rsi_value:.2f}")
            print(rsi_interpretation)
        else:
            print(rsi_interpretation)
        
        print("\n # MACD (Moving Average Convergence Divergence) Analysis:")
        macd_values, macd_interpretation = results['macd_data']
        if macd_values is not None:
            macd, signal, histogram = macd_values
            print(f"|MACD: {macd:.4f}")
            print(f"|Signal Line: {signal:.4f}")
            print(f"|Histogram: {histogram:.4f}")
            print(macd_interpretation)
        else:
            print(macd_interpretation)

        print("\n # Supply Distribution Analysis:")
        supply_distribution_data = results['supply_distribution_data']
        if supply_distribution_data:
            print(self.analyze_supply_distribution(supply_distribution_data))
        else:
            print("Unable to obtain supply distribution data.")    

        print("\n # EPR (Estimated Price Ratio) Analysis:")
        epr_data = results['epr_data']
        if epr_data is not None:
            print(self.analyze_epr(epr_data))
        else:
            print("Unable to calculate EPR for analysis.")

        print("\n # NVT Ratio (Network Value to Transactions Ratio) Analysis:")
        nvt_data = results['nvt_data']
        if nvt_data is not None:
            print(self.analyze_nvt_ratio(nvt_data))
        else:
            print("Unable to calculate NVT Ratio for analysis.")

        print("\n # Network Hash Rate Analysis:")
        hashrate_data = results['hashrate_data']
        if hashrate_data is not None:
            print(self.analyze_network_hashrate(hashrate_data))
        else:
            print("Unable to obtain network hash rate data for analysis.")

        print("\n # Global Interest Rates Macroeconomic Analysis:")
        global_rates_data = results.get('global_rates_data')
        if global_rates_data is not None:
            print(self.analyze_global_interest_rates(global_rates_data))
        else:
            print("Unable to obtain complete global interest rates data.")
        
        print("\n # Transaction Fees Analysis:")
        transaction_fees = results.get('transaction_fees')
        if transaction_fees:
            print(self.analyze_transaction_fees(transaction_fees))
        else:
            print("Unable to obtain transaction fees data.")
        
        print("\n # Recent Blocks Analysis:")
        recent_blocks = results.get('recent_blocks')
        if recent_blocks:
            print(self.analyze_recent_blocks(recent_blocks))
        else:
            print("Unable to obtain recent blocks data.")

        print("\n # Mempool Status Analysis:")
        mempool_status = results.get('mempool_status')
        if mempool_status:
            print(self.analyze_mempool(mempool_status))
        else:
            print("Unable to obtain mempool status data.")

        print("\n # Mining Difficulty Analysis:")
        mining_difficulty = results.get('mining_difficulty')
        if mining_difficulty:
            print(self.analyze_mining_difficulty(mining_difficulty))
        else:
            print("Unable to obtain mining difficulty data.")

        print("\n # Bitcoin Dominance Analysis:")
        bitcoin_dominance = results.get('bitcoin_dominance')
        if bitcoin_dominance is not None:
            print(self.analyze_bitcoin_dominance(bitcoin_dominance))
        else:
            print("Unable to obtain Bitcoin dominance data.")
    
        print("\n \n ## FINAL EVALUATION:")
        recommendation, score = self.get_master_evaluation(results)
        print(f"|Recommendation: {recommendation}")
        print(f"|Confidence Score: {abs(score):.2f}")
        
        if recommendation == "BUY":
            print("The indicators suggest an upward trend. Consider BUYING, but always do your own research.")
        elif recommendation == "SELL":
            print("The indicators suggest a downward trend. Consider SELLING, but always do your own research.")
        else:
            print("The indicators are mixed. Consider HOLDING your current position and monitor closely.")
        
        print("\nRemember: This is an automated analysis and does not replace professional financial advice.")
        print("Always do your own research and consider your risk tolerance before making investment decisions.")
        
        print("="*50)

    def run(self):
        print("\nWelcome to the Enhanced Bitcoin Market Analyzer :)")
        while True:
            try:
                print("\nWhat would you like to do?")
                print("1. Analyze the Bitcoin market")
                print("2. Clear data cache")
                print("3. Exit")
                
                choice = input("\nEnter your choice (1-3): ")
                
                if choice == '1':
                    self.analyze()
                elif choice == '2':
                    if os.path.exists(self.cache_file):
                        os.remove(self.cache_file)
                        self.cache = {}
                        print("\nData cache cleared successfully.")
                        self.logger.info("\nData cache cleared by the user.")
                    else:
                        print("No data cache to clear.")
                elif choice == '3':
                    print("\nThank you for using the Enhanced Bitcoin Market Analyzer. Goodbye!\n")
                    self.logger.info("Analysis session ended by the user.")
                    break
                else:
                    print("\nInvalid choice. Please enter 1, 2, or 3.")
            except Exception as e:
                self.logger.error(f"\nUnexpected error: {str(e)}")
                print("\nAn unexpected error occurred. Please try again.")

            print("\n" + "-"*50)

class GoogleTrendsAnalyzer:
    def __init__(self):
        self.keywords = [
            "bitcoin", "crypto", "blockchain", "btc price", "bitcoin trading",
            "bitcoin investment", "cryptocurrency market", "bitcoin news",
            "bitcoin halving", "bitcoin wallet", "bitcoin mining"
        ] 
        self.regions = ["US", "GB", "JP", "KR", "DE", "NG"]  # Major cryptocurrency markets
        self.url = "https://trends.google.com/trends/api/dailytrends"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.session = requests.Session()
        self.max_retries = 5
        self.base_delay = 5  # seconds

    def get_trends_data(self, region):
        params = {
            "hl": "en-US",
            "tz": "-180",
            "geo": region,
            "ns": "15"
        }
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(self.url, params=params, headers=self.headers, timeout=30)
                response.raise_for_status()
                data = response.text[5:]  # Remove ")]}',\n" at the beginning
                return json.loads(data)
            except requests.RequestException as e:
                if response.status_code == 429:
                    delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limit reached for {region}. Attempt {attempt+1}/{self.max_retries}. Waiting {delay:.2f} seconds.")
                    time.sleep(delay)
                else:
                    logger.error(f"Error fetching Google Trends data for {region}: {e}")
                    break
        logger.error(f"Failed to obtain Google Trends data for {region} after {self.max_retries} attempts.")
        return None

    def analyze_sentiment(self, title):
        blob = TextBlob(title)
        return blob.sentiment.polarity

    def calculate_relevance(self, title):
        return sum(keyword in title.lower() for keyword in self.keywords) / len(self.keywords)

    def process_trends(self, trends_data):
        if not trends_data:
            return 0, 0

        total_score = 0
        total_volume = 0

        for trend in trends_data['default']['trendingSearchesDays'][0]['trendingSearches']:
            title = trend['title']['query'].lower()
            traffic = int(trend['formattedTraffic'].replace('+', '').replace('K', '000'))
            
            relevance = self.calculate_relevance(title)
            sentiment = self.analyze_sentiment(title)
            
            trend_score = traffic * relevance * (sentiment + 1)  # Normalize sentiment to 0-2 range
            
            total_score += trend_score
            total_volume += traffic

        return total_score, total_volume

    def get_historical_average(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        historical_scores = []
        
        for date in pd.date_range(start=start_date, end=end_date):
            score = self.get_daily_score(date)
            if score is not None:
                historical_scores.append(score)
        
        if not historical_scores:
            logger.warning("Unable to obtain historical scores")
            return 1  # Return 1 to avoid division by zero
        
        return np.mean(historical_scores)

    def get_daily_score(self, date):
        daily_score = 0
        daily_volume = 0
        
        for region in self.regions:
            trends_data = self.get_trends_data(region)
            if trends_data:
                score, volume = self.process_trends(trends_data)
                daily_score += score
                daily_volume += volume
        
        return daily_score / daily_volume if daily_volume > 0 else None

    def get_google_trends_data(self):
        try:
            logger.info("\nAnalyzing Google Trends data...")
            historical_average = self.get_historical_average()
            
            total_score = 0
            total_volume = 0
            successful_regions = 0
            
            for region in self.regions:
                trends_data = self.get_trends_data(region)
                if trends_data:
                    score, volume = self.process_trends(trends_data)
                    total_score += score
                    total_volume += volume
                    successful_regions += 1
            
            if successful_regions == 0:
                logger.warning("Unable to obtain data from any region. Returning neutral score.")
                return 50  # Neutral score
            
            current_score = total_score / total_volume if total_volume > 0 else 0
            normalized_score = (current_score / historical_average) * 50 if historical_average > 0 else 50
            
            # Ensure the score is within the 0-100 range
            final_score = max(0, min(100, normalized_score))
            
            logger.info(f"\nGoogle Trends analysis completed. Successful regions: {successful_regions}/{len(self.regions)}")
            return final_score
        
        except Exception as e:
            self.logger.error("\nError in Google Trends analysis")
            return 50  # Return a neutral score in case of error

def main():
    if os.name == 'nt':  # For Windows
        os.system('title Bitcoin Market Analyzer')
    else:  # For Unix/Linux/macOS
        os.system('echo -ne "\033]0;Bitcoin Market Analyzer\007"')

    analyzer = EnhancedBitcoinAnalyzer()
    analyzer.run()
    
if __name__ == "__main__":
    main()

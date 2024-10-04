# CÓDIGO POR: RODRIGO S MAGALHÃES (https://github.com/FDBnet)

# AVISOS:
"""
    Há dois principais BUGS:
    - A obtenção da Taxa de Juros através do 'Bank of Japan' e 'Bank of England' não está funcionando como esperado;
    - A obtenção do Hash Rate da rede Bitcoin também não está funcionando!
"""
# SUBSTITUA "SUA_CHAVE_API_AQUI" POR UMA API REAL. CONSIGA UMA EM "https://fredaccount.stlouisfed.org/apikey"

# CONTRIBUA COM O PROJETO DOANDO BITCOIN OU SATOSHIS PARA: bc1qcgxvxp0v9gtac8srkl7rkrvflfdkmtasu35txv

# VERSÃO EM PORTUGUÊS DO BRASIL

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

# Configuração de logging personalizada
class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.INFO:
            return f"\n{record.msg}"
        elif record.levelno == logging.WARNING:
            return f"\nAviso: {record.msg}"
        elif record.levelno == logging.ERROR:
            return f"\nErro: {record.msg}"
        return super().format(record)

# Configuração do logger
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
                self.logger.warning(f"Tentativa {attempt + 1} falhou para {url}")
                if attempt == max_retries - 1:
                    self.logger.error(f"Falha ao obter dados após {max_retries} tentativas.")
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
            self.logger.error(f"Erro ao fazer requisição para {url}: {str(e)}")
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
            return "Não foi possível comparar devido a dados insuficientes."
        
        if current_price > ma_200:
            return f"|O preço atual (${current_price:.2f}) é maior que a média móvel de 200 dias (${ma_200:.2f})."
        else:
            return f"|O preço atual (${current_price:.2f}) é menor ou igual à média móvel de 200 dias (${ma_200:.2f})."

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
        estimated_funding_rate = price_difference * 3 * 100  # Convertido para porcentagem
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
                    logger.warning(f"Falha ao obter Funding Rate de {exchange}.")
                    return None
            try:
                return float(data) * 100  # Convertendo para porcentagem
            except (ValueError, TypeError):
                logger.warning(f"Falha ao converter Funding Rate de {exchange}.")
                return None
        logger.error(f"Não foi possível obter dados de {exchange}.")
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
            
            self.logger.warning("Falha ao obter taxa de financiamento da Bybit v5.")
            return None
        except Exception as e:
            self.logger.error(f"Erro ao obter taxa de financiamento da Bybit: {str(e)}")
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
        
        return rsi.iloc[-1]  # Retorna o RSI mais recente

    def get_rsi_evaluation(self):
        try:
            df = self.get_historical_data(days=30)  # Pegamos 30 dias de dados para calcular o RSI
            if df.empty:
                logger.warning("Dados insuficientes para calcular o RSI")
                return None, "Dados insuficientes para calcular o RSI"
            
            rsi = self.calculate_rsi(df)
            
            if rsi > 70:
                return rsi, "O mercado pode estar sobrecomprado. Considere a possibilidade de venda ou tome cuidado ao comprar."
            elif rsi < 30:
                return rsi, "O mercado pode estar sobrevendido. Pode ser uma oportunidade de compra, mas esteja atento a outros indicadores."
            else:
                return rsi, "O RSI está em uma faixa neutra. Considere outros indicadores para tomar decisões."
        
        except Exception as e:
            logger.error(f"Erro ao calcular o RSI: {e}")
            return None, f"Não foi possível calcular o RSI devido a um erro: {e}"

    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        try:
            # Calcula as médias móveis exponenciais
            ema_fast = data['price'].ewm(span=fast_period, adjust=False).mean()
            ema_slow = data['price'].ewm(span=slow_period, adjust=False).mean()
            
            # Calcula a linha MACD
            macd_line = ema_fast - ema_slow
            
            # Calcula a linha de sinal
            signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
            
            # Calcula o histograma
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.error(f"Erro ao calcular o MACD: {e}")
            return None, None, None

    def get_macd_evaluation(self):
        try:
            df = self.get_historical_data(days=60)  # Pegamos 60 dias de dados para calcular o MACD
            if df.empty:
                logger.warning("Dados insuficientes para calcular o MACD")
                return None, "Dados insuficientes para calcular o MACD"
            
            macd_line, signal_line, histogram = self.calculate_macd(df)
            
            if macd_line is None or signal_line is None:
                return None, "Não foi possível calcular o MACD"
            
            # Pegamos os últimos valores para a comparação
            last_macd = macd_line.iloc[-1]
            last_signal = signal_line.iloc[-1]
            last_histogram = histogram.iloc[-1]
            prev_histogram = histogram.iloc[-2]
            
            if last_macd > last_signal:
                if last_histogram > 0 and last_histogram > prev_histogram:
                    interpretation = "O MACD está acima da linha de sinal e o histograma está aumentando. Isso pode indicar um forte momentum de alta."
                else:
                    interpretation = "O MACD está acima da linha de sinal. Isso pode indicar uma tendência de alta, mas observe o histograma para confirmação."
            elif last_macd < last_signal:
                if last_histogram < 0 and last_histogram < prev_histogram:
                    interpretation = "O MACD está abaixo da linha de sinal e o histograma está diminuindo. Isso pode indicar um forte momentum de baixa."
                else:
                    interpretation = "O MACD está abaixo da linha de sinal. Isso pode indicar uma tendência de baixa, mas observe o histograma para confirmação."
            else:
                interpretation = "O MACD e a linha de sinal estão próximos. O mercado pode estar em um ponto de indecisão."
            
            return (last_macd, last_signal, last_histogram), interpretation
        
        except Exception as e:
            logger.error(f"Erro ao avaliar o MACD: {e}")
            return None, f"Não foi possível avaliar o MACD devido a um erro: {e}"

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
                logger.warning("Não foi possível obter dados de distribuição de oferta")
                return None

            supply_distribution = {
                'circulante': data['market_data']['circulating_supply'],
                'total': data['market_data']['total_supply'],
                'max': data['market_data']['max_supply']
            }

            return supply_distribution

        except Exception as e:
            logger.error(f"Erro ao obter dados de distribuição de oferta: {e}")
            return None

    def analyze_supply_distribution(self, supply_distribution):
        if not supply_distribution:
            return "Não foi possível analisar a distribuição de oferta devido à falta de dados."

        circulante = supply_distribution['circulante']
        total = supply_distribution['total']
        max_supply = supply_distribution['max']

        percent_circulante = (circulante / max_supply) * 100
        percent_nao_circulante = ((total - circulante) / max_supply) * 100
        percent_nao_minerado = ((max_supply - total) / max_supply) * 100

        analysis = f"| {percent_circulante:.2f}% do supply máximo está em circulação.\n"
        analysis += f"| {percent_nao_circulante:.2f}% do supply máximo foi minerado mas não está em circulação.\n"
        analysis += f"| {percent_nao_minerado:.2f}% do supply máximo ainda não foi minerado.\n"

        if percent_circulante > 90:
            analysis += "Alta proporção de oferta em circulação, o que pode indicar ampla distribuição.\n"
        elif percent_nao_circulante > 10:
            analysis += "Significativa proporção de oferta não circulante, o que pode indicar acumulação.\n"

        return analysis

    def calculate_epr(self, price_data, window=7):
        """
        Calcula o Estimated Price Ratio (EPR), uma aproximação do SOPR.
        
        :param price_data: DataFrame com os preços históricos
        :param window: Janela para cálculo da média móvel (em dias)
        :return: Series com o EPR calculado
        """
        # Calcula o retorno diário
        daily_return = price_data['price'].pct_change()
        
        # Calcula a média móvel dos retornos
        moving_avg_return = daily_return.rolling(window=window).mean()
        
        # Calcula o EPR
        epr = (1 + daily_return) / (1 + moving_avg_return)
        
        return epr

    def analyze_epr(self, epr_data):
        if epr_data is None or epr_data.empty:
            return "Dados EPR insuficientes para análise."
        
        latest_epr = epr_data.iloc[-1]
        avg_epr = epr_data.mean()
        
        analysis = f"|EPR atual: {latest_epr:.4f}\n"
        analysis += f"|EPR médio (7 dias): {avg_epr:.4f}\n"
        
        if latest_epr > 1:
            if latest_epr > avg_epr:
                analysis += "O EPR está acima de 1 e acima da média, sugerindo que os preços recentes estão acima da tendência de curto prazo. Isso pode indicar um momento de alta no mercado."
            else:
                analysis += "O EPR está acima de 1 mas próximo da média, sugerindo um equilíbrio entre otimismo e realização de lucros."
        else:
            if latest_epr < avg_epr:
                analysis += "O EPR está abaixo de 1 e abaixo da média, sugerindo que os preços recentes estão abaixo da tendência de curto prazo. Isso pode indicar um momento de baixa no mercado."
            else:
                analysis += "O EPR está abaixo de 1 mas próximo da média, sugerindo que o mercado pode estar se aproximando de um ponto de virada."
        
        return analysis
    
    def calculate_nvt_ratio(self, days=30):
        """
        Calcula uma aproximação do NVT Ratio usando capitalização de mercado e volume de negociação.
        
        :param days: Número de dias para coletar dados
        :return: DataFrame com o NVT Ratio calculado
        """
        url = f"{self.coingecko_base_url}/coins/{self.symbol}/market_chart"
        params = {
            "vs_currency": self.vs_currency,
            "days": days
        }
        
        data = self.rate_limited_request(url, params)
        if not data:
            logger.warning("Não foi possível obter dados para o cálculo do NVT Ratio")
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
        
        # Calcula o NVT Ratio
        df['nvt_ratio'] = df['market_cap'] / df['volume']
        
        return df

    def analyze_nvt_ratio(self, nvt_data):
        if nvt_data is None or nvt_data.empty:
            return "Dados NVT insuficientes para análise."
        
        latest_nvt = nvt_data['nvt_ratio'].iloc[-1]
        avg_nvt = nvt_data['nvt_ratio'].mean()
        
        analysis = f"|NVT Ratio atual: {latest_nvt:.2f}\n"
        analysis += f"|NVT Ratio médio (30 dias): {avg_nvt:.2f}\n"
        
        if latest_nvt > avg_nvt * 1.5:
            analysis += "O NVT Ratio está significativamente acima da média, sugerindo que o valor da rede pode estar sobrevalorizado em relação à atividade econômica."
        elif latest_nvt > avg_nvt * 1.2:
            analysis += "O NVT Ratio está acima da média, indicando uma possível sobrevalorização, mas ainda dentro de limites razoáveis."
        elif latest_nvt < avg_nvt * 0.8:
            analysis += "O NVT Ratio está abaixo da média, sugerindo que o valor da rede pode estar subvalorizado em relação à atividade econômica."
        else:
            analysis += "O NVT Ratio está próximo da média, indicando uma valorização equilibrada em relação à atividade econômica."
        
        return analysis
    
    def get_network_hashrate(self, days=30):
        """
        Obtém dados da hash rate da rede Bitcoin usando a API do mempool.space.
        
        :param days: Número de dias para coletar dados
        :return: DataFrame com os dados da hash rate
        """
        endpoint = f"v1/mining/hashrate/{days * 144}"  # 144 blocos por dia em média
        data = self.mempool_api_request(endpoint)
        
        if not data:
            self.logger.warning("Não foi possível obter dados da hash rate da rede")
            return None

        df = pd.DataFrame(data, columns=['timestamp', 'hashrate'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        df['hashrate'] = df['hashrate'] / 1e9  # Convertendo para EH/s
        
        return df
    
    def get_transaction_fees(self):
        """
        Obtém dados sobre as taxas de transação atuais na rede Bitcoin.
        
        :return: Dict com informações sobre as taxas de transação
        """
        endpoint = "v1/fees/recommended"
        data = self.mempool_api_request(endpoint)
        
        if not data:
            self.logger.warning("Não foi possível obter dados de taxas de transação")
            return None

        return data

    def analyze_network_hashrate(self, hashrate_data):
        if hashrate_data is None or hashrate_data.empty:
            return "Dados de hash rate insuficientes para análise."
        
        latest_hashrate = hashrate_data['hashrate'].iloc[-1]
        avg_hashrate = hashrate_data['hashrate'].mean()
        
        if np.isnan(latest_hashrate) or np.isnan(avg_hashrate):
            return "Não foi possível calcular a hash rate devido a dados inválidos."
        
        pct_change = ((latest_hashrate - hashrate_data['hashrate'].iloc[0]) / hashrate_data['hashrate'].iloc[0]) * 100
        
        analysis = f"|Hash Rate atual: {latest_hashrate:.2f} EH/s\n"
        analysis += f"|Hash Rate média (30 dias): {avg_hashrate:.2f} EH/s\n"
        analysis += f"|Variação percentual (30 dias): {pct_change:.2f}%\n"
        
        if latest_hashrate > avg_hashrate * 1.1:
            analysis += "A hash rate atual está significativamente acima da média, indicando um aumento na segurança da rede e possivelmente maior interesse dos mineradores."
        elif latest_hashrate < avg_hashrate * 0.9:
            analysis += "A hash rate atual está abaixo da média, o que pode indicar uma redução na atividade de mineração ou mudanças nas condições do mercado."
        else:
            analysis += "A hash rate atual está próxima da média, sugerindo estabilidade na atividade de mineração."
        
        if pct_change > 10:
            analysis += "\nO aumento significativo na hash rate pode indicar uma maior confiança na rede e potencialmente um sinal bullish para o preço."
        elif pct_change < -10:
            analysis += "\nA diminuição significativa na hash rate pode indicar desafios para os mineradores ou mudanças regulatórias, potencialmente um sinal bearish."
        
        return analysis
    
    def get_global_interest_rates(self):
        interest_rates = {
            '- FED (EUA)': self.get_fed_rate(),
            '- BCE (Europa)': self.get_ecb_rate(),
            '- BoJ (Japão)': self.get_boj_rate(),
            '- BoE (Reino Unido)': self.get_boe_rate()
        }
        
        # Remove None values
        interest_rates = {k: v for k, v in interest_rates.items() if v is not None}
        
        if not interest_rates:
            self.logger.warning("Não foi possível obter nenhuma taxa de juros global.")
            return None
        
        return pd.DataFrame(list(interest_rates.items()), columns=['Central Bank', 'Interest Rate'])

    def get_fed_rate(self):
        try:
            # Método primário: scraping do site do Federal Reserve
            url = "https://www.federalreserve.gov/releases/h15/"
            response = self.rate_limited_request(url)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                rate_element = soup.find('th', id='id94d1cc0', string='Federal funds (effective)')
                if rate_element:
                    rate = rate_element.find_next('td', class_='data').text
                    return float(rate.strip())
            
            # Método secundário: API do FRED (Federal Reserve Economic Data)
            fred_url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": "FEDFUNDS",
                "api_key": "SUA_CHAVE_API_AQUI",
                "sort_order": "desc",
                "limit": 1,
                "file_type": "json"
            }
            fred_data = self.rate_limited_request(fred_url, params)
            if fred_data and 'observations' in fred_data:
                return float(fred_data['observations'][0]['value'])
            
            self.logger.error("Falha ao obter a taxa do FED de todas as fontes.")
            return None
        except Exception as e:
            self.logger.error(f"Erro ao obter a taxa do FED: {str(e)}")
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
            self.logger.error("Não foi possível encontrar as taxas do BCE na página.")
            return None
        except Exception as e:
            self.logger.error(f"Erro ao obter taxa do BCE: {str(e)}")
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
            self.logger.error(f"Erro ao obter taxa do BOJ: {str(e)}")
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
            self.logger.warning("Não foi possível obter a taxa do Bank of England.")
            return None
        except Exception as e:
            self.logger.error(f"Erro ao obter taxa BOE: {str(e)}")
            return None

    def analyze_global_interest_rates(self, rates_data):
        if rates_data is None or rates_data.empty:
            return "Dados de taxas de juros globais insuficientes para análise."
        
        analysis = ""
        for _, row in rates_data.iterrows():
            analysis += f"{row['Central Bank']}: {row['Interest Rate']:.2f}%\n"
        
        avg_rate = rates_data['Interest Rate'].mean()
        analysis += f"\n|Taxa média global: {avg_rate:.2f}%\n"
        
        if avg_rate < 1:
            analysis += "\nAs taxas de juros globais estão muito baixas, o que geralmente é favorável para ativos de risco como o Bitcoin."
        elif avg_rate < 3:
            analysis += "\nAs taxas de juros globais estão moderadas, o que pode ainda ser favorável para o Bitcoin, mas com menos intensidade."
        else:
            analysis += "\nAs taxas de juros globais estão relativamente altas, o que pode pressionar ativos de risco como o Bitcoin."
        
        return analysis
    
    def analyze_transaction_fees(self, fee_data):
        if fee_data is None:
            return "Dados de taxas de transação insuficientes para análise."
        
        analysis = ""
        analysis += f"|Taxa rápida (próximo bloco): {fee_data['fastestFee']} sat/vB\n"
        analysis += f"|Taxa média (meia hora): {fee_data['halfHourFee']} sat/vB\n"
        analysis += f"|Taxa econômica (1 hora): {fee_data['hourFee']} sat/vB\n"
        
        if fee_data['fastestFee'] > 100:
            analysis += "\nAs taxas de transação estão muito altas, indicando alta demanda na rede."
        elif fee_data['fastestFee'] > 50:
            analysis += "\nAs taxas de transação estão moderadamente altas, sugerindo demanda significativa."
        elif fee_data['fastestFee'] < 10:
            analysis += "\nAs taxas de transação estão baixas, indicando pouca congestionamento na rede."
        else:
            analysis += "\nAs taxas de transação estão em níveis normais."
        
        return analysis
    
    def get_recent_blocks(self, max_retries=5, delay=5):
        """
        Obtém dados dos blocos mais recentes da API pública do mempool.space
        
        :param max_retries: Número máximo de tentativas
        :param delay: Tempo de espera entre tentativas (em segundos)
        :return: Lista de dicionários com dados dos blocos ou None em caso de falha
        """
        url = "https://mempool.space/api/v1/blocks"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                self.logger.warning(f"Tentativa {attempt + 1} falhou: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))
        
        self.logger.error(f"Falha ao obter dados dos blocos recentes após {max_retries} tentativas")
        return None

    def analyze_recent_blocks(self, blocks_data):
        if not blocks_data:
            return "Dados de blocos recentes indisponíveis para análise."
        
        analysis = "Análise de Blocos Recentes:\n"
        avg_block_size = sum(block['size'] for block in blocks_data) / len(blocks_data)
        avg_tx_count = sum(block['tx_count'] for block in blocks_data) / len(blocks_data)
        
        analysis += f"|Tamanho médio dos blocos: {avg_block_size / 1000:.2f} KB\n"
        analysis += f"|Média de transações por bloco: {avg_tx_count:.0f}\n"
        
        if avg_block_size > 1_300_000:  # 1.3 MB
            analysis += "Os blocos estão quase cheios, indicando alta demanda na rede.\n"
        elif avg_block_size < 500_000:  # 500 KB
            analysis += "Os blocos estão relativamente vazios, sugerindo baixa demanda na rede.\n"
        else:
            analysis += "O tamanho dos blocos está em níveis normais.\n"
        
        return analysis
    
    def get_mempool_data(self, max_retries=5, delay=5):
        """
        Obtém dados do mempool da API pública do mempool.space
        
        :param max_retries: Número máximo de tentativas
        :param delay: Tempo de espera entre tentativas (em segundos)
        :return: Dicionário com dados do mempool ou None em caso de falha
        """
        url = "https://mempool.space/api/v1/mempool"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                self.logger.warning(f"Tentativa {attempt + 1} falhou: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))
        
        self.logger.error(f"Falha ao obter dados do mempool após {max_retries} tentativas")
        return None

    def analyze_mempool(self, mempool_data):
        if not mempool_data:
            return "Dados do mempool indisponíveis para análise."
        
        analysis = "Análise do Mempool:\n"
        analysis += f"|Tamanho do mempool: {mempool_data['vsize'] / 1_000_000:.2f} MB\n"
        analysis += f"|Número de transações: {mempool_data['count']}\n"
        
        if mempool_data['vsize'] > 80_000_000:  # 80 MB
            analysis += "O mempool está muito congestionado. Espere taxas de transação altas.\n"
        elif mempool_data['vsize'] < 5_000_000:  # 5 MB
            analysis += "O mempool está relativamente vazio. As taxas de transação devem estar baixas.\n"
        else:
            analysis += "O mempool está em níveis normais.\n"
        
        return analysis
    
    def get_mining_difficulty(self, max_retries=5, delay=5):
        """
        Obtém dados de dificuldade de mineração da API pública do mempool.space
        
        :param max_retries: Número máximo de tentativas
        :param delay: Tempo de espera entre tentativas (em segundos)
        :return: Dicionário com dados de dificuldade ou None em caso de falha
        """
        url = "https://mempool.space/api/v1/difficulty-adjustment"
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                self.logger.warning(f"Tentativa {attempt + 1} falhou: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (2 ** attempt))
        
        self.logger.error(f"Falha ao obter dados de dificuldade após {max_retries} tentativas")
        return None

    def analyze_mining_difficulty(self, difficulty_data):
        if not difficulty_data:
            return "Dados de dificuldade de mineração indisponíveis para análise."
        
        analysis = "Análise da Dificuldade de Mineração:\n"
        
        try:
            current_difficulty = difficulty_data.get('current_difficulty')
            if current_difficulty is not None:
                analysis += f"|Dificuldade atual: {current_difficulty:,}\n"
            else:
                analysis += "Dificuldade atual não disponível.\n"
            
            estimated_retarget = difficulty_data.get('estimated_retarget_percentage')
            if estimated_retarget is not None:
                analysis += f"|Estimativa de mudança: {estimated_retarget:.2f}%\n"
            else:
                analysis += "Estimativa de mudança não disponível.\n"
            
            remaining_blocks = difficulty_data.get('remaining_blocks')
            if remaining_blocks is not None:
                analysis += f"|Blocos até o ajuste: {remaining_blocks}\n"
            else:
                analysis += "Número de blocos até o ajuste não disponível.\n"
            
            if estimated_retarget is not None:
                if estimated_retarget > 5:
                    analysis += "A dificuldade provavelmente aumentará, indicando aumento na capacidade de mineração.\n"
                elif estimated_retarget < -5:
                    analysis += "A dificuldade provavelmente diminuirá, possivelmente indicando redução na capacidade de mineração.\n"
                else:
                    analysis += "A dificuldade deve permanecer relativamente estável.\n"
        except Exception as e:
            self.logger.error(f"Erro ao analisar dificuldade de mineração: {str(e)}")
            analysis += "Ocorreu um erro ao analisar os dados de dificuldade de mineração.\n"
        
        return analysis

    def get_bitcoin_dominance(self):
        """
        Obtém a dominância atual do Bitcoin no mercado de criptomoedas.
        
        :return: Float representando a porcentagem de dominância do Bitcoin ou None se não for possível obter os dados
        """
        url = "https://api.coingecko.com/api/v3/global"
        try:
            data = self.rate_limited_request(url)
            
            if data and isinstance(data, dict) and 'data' in data:
                market_data = data['data']
                if 'market_cap_percentage' in market_data and 'btc' in market_data['market_cap_percentage']:
                    return market_data['market_cap_percentage']['btc']
            
            self.logger.warning("Não foi possível obter dados de dominância do Bitcoin da API")
            return None
        except Exception as e:
            self.logger.error(f"Erro ao obter dominância do Bitcoin: {str(e)}")
            return None

    def analyze_bitcoin_dominance(self, dominance):
        if dominance is None:
            return "Dados de dominância do Bitcoin insuficientes para análise."
        
        analysis = f"Dominância do Bitcoin: {dominance:.2f}%\n"
        
        if dominance > 60:
            analysis += "A alta dominância do Bitcoin sugere forte confiança no BTC em relação a outras criptomoedas."
        elif dominance < 40:
            analysis += "A baixa dominância do Bitcoin pode indicar um aumento no interesse por altcoins ou uma diminuição na confiança no BTC."
        else:
            analysis += "A dominância do Bitcoin está em níveis moderados, sugerindo um equilíbrio entre BTC e altcoins."
        
        return analysis

    def analyze(self):
        self.logger.info("\nIniciando análise do mercado de Bitcoin...")
        print("\nAnalisando o mercado de Bitcoin...", end="", flush=True)
        
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
                    self.logger.error(f"Erro ao obter {futures[future]}: {str(e)}")
                    results[futures[future]] = None
                    print("x", end="", flush=True)

        print("\nAnálise concluída!")

        # Calcula o EPR
        price_data_for_epr = results.get('price_data_for_epr')
        if price_data_for_epr is not None and not price_data_for_epr.empty:
            epr_data = self.calculate_epr(price_data_for_epr)
            results['epr_data'] = epr_data
        else:
            results['epr_data'] = None

        self.print_analysis_results(results)

    def get_master_evaluation(self, results):
        try:
            price_trend = 1 if "maior que" in results.get('price_comparison', '') else -1
            
            funding_rates = [
                results.get('estimated_funding_rate'),
                results.get('binance_rate'),
                results.get('bybit_rate'),
                results.get('okex_rate')
            ]
            valid_rates = [r for r in funding_rates if r is not None]
            avg_funding_rate = np.mean(valid_rates) if valid_rates else 0
            
            sentiment_score = results.get('sentiment_score', 50)  # Valor neutro se não disponível
            
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
                percent_circulante = (supply_distribution_data['circulante'] / supply_distribution_data['max']) * 100
                if percent_circulante > 90:
                    supply_distribution_score = 1  # Sinal positivo (ampla distribuição)
                elif percent_circulante < 80:
                    supply_distribution_score = -1  # Sinal negativo (possível concentração)
            
            epr_data = results['epr_data']
            epr_score = 0
            if epr_data is not None and not epr_data.empty:
                latest_epr = epr_data.iloc[-1]
                if latest_epr > 1:
                    epr_score = 1  # Possível momento de alta
                elif latest_epr < 1:
                    epr_score = -1   # Possível momento de baixa
            
            nvt_data = results['nvt_data']
            nvt_score = 0
            if nvt_data is not None and not nvt_data.empty:
                latest_nvt = nvt_data['nvt_ratio'].iloc[-1]
                avg_nvt = nvt_data['nvt_ratio'].mean()
                if latest_nvt > avg_nvt * 1.2:
                    nvt_score = -1  # Possível sobrevalorização
                elif latest_nvt < avg_nvt * 0.8:
                    nvt_score = 1   # Possível subvalorização
            
            hashrate_data = results['hashrate_data']
            hashrate_score = 0
            if hashrate_data is not None and not hashrate_data.empty:
                latest_hashrate = hashrate_data['hashrate'].iloc[-1]
                avg_hashrate = hashrate_data['hashrate'].mean()
                pct_change = ((latest_hashrate - hashrate_data['hashrate'].iloc[0]) / hashrate_data['hashrate'].iloc[0]) * 100
                
                if latest_hashrate > avg_hashrate * 1.1 or pct_change > 10:
                    hashrate_score = 1  # Sinal positivo
                elif latest_hashrate < avg_hashrate * 0.9 or pct_change < -10:
                    hashrate_score = -1  # Sinal negativo
            
            global_rates_data = results['global_rates_data']
            global_rates_score = 0
            if global_rates_data is not None and not global_rates_data.empty:
                avg_rate = global_rates_data['Interest Rate'].mean()
                if avg_rate < 1:
                    global_rates_score = 1  # Favorável para Bitcoin
                elif avg_rate > 3:
                    global_rates_score = -1  # Menos favorável para Bitcoin
            
            # Ajuste os pesos para incluir as Taxas de Juros Globais
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
            
            # Cálculo do score final (incluindo Taxas de Juros Globais)
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
                return "COMPRAR", final_score
            elif final_score < -0.5:
                return "VENDER", final_score
            else:
                return "MANTER", final_score
        except Exception as e:
            self.logger.error(f"Erro na avaliação final: {str(e)}")
            return "INCONCLUSIVO", 0

    def print_analysis_results(self, results):
        print("\n" + "="*50)
        print(" ## ANÁLISE DE MERCADO DO BITCOIN ##")
        print("="*50 + "\n")
        print(results['price_comparison'])
        print("\n # Funding Rates:")
        print(f"{'Exchange':<12} {'Rate':>12}")
        print("-"*22)
        for exchange, rate_key in [("- Estimativa", "estimated_funding_rate"), 
                                   ("- Binance", "binance_rate"), 
                                   ("- Bybit", "bybit_rate"), 
                                   ("- OKX", "okex_rate")]:
            if results[rate_key] is not None:
                print(f"{exchange:<10} {results[rate_key]:>10.4f}%")
        
        print("\nNota: Estas são taxas em tempo real e podem não refletir")
        print("o Funding Rate exato para o próximo período.")
        
        valid_rates = [r for r in [results['binance_rate'], results['bybit_rate'], results['okex_rate']] if r is not None]
        if valid_rates:
            avg_rate = np.mean(valid_rates)
            print(f"\n|Média das Funding Rates: {avg_rate:.4f}%")
            if abs(avg_rate) > 0.1:  # 0.1%
                print("Atenção: Média das Funding Rates está elevada,")
                print("indicando possível volatilidade no mercado.")
            elif abs(avg_rate) < 0.01:  # 0.01%
                print("Média das Funding Rates está em níveis baixos,")
                print("indicando possível estabilidade no mercado.")
            else:
                print("Média das Funding Rates está em níveis moderados.")
        else:
            print("Não foi possível calcular a média das Funding Rates")
            print("devido à falta de dados.")
        
        print("\n # Análise de Sentimento do Google Trends:")
        sentiment_score = results['sentiment_score']
        print(f"|Score de Sentimento: {sentiment_score:.2f}")
        if sentiment_score > 70:
            print("O sentimento do mercado está muito positivo.")
        elif sentiment_score > 50:
            print("O sentimento do mercado está levemente positivo.")
        elif sentiment_score > 30:
            print("O sentimento do mercado está neutro.")
        else:
            print("O sentimento do mercado está negativo.")
        
        print("\n # Análise do RSI (Índice de Força Relativa):")
        rsi_value, rsi_interpretation = results['rsi_data']
        if rsi_value is not None:
            print(f"|RSI atual: {rsi_value:.2f}")
            print(rsi_interpretation)
        else:
            print(rsi_interpretation)
        
        print("\n # Análise do MACD (Convergência e Divergência de Médias Móveis):")
        macd_values, macd_interpretation = results['macd_data']
        if macd_values is not None:
            macd, signal, histogram = macd_values
            print(f"|MACD: {macd:.4f}")
            print(f"|Linha de Sinal: {signal:.4f}")
            print(f"|Histograma: {histogram:.4f}")
            print(macd_interpretation)
        else:
            print(macd_interpretation)

        print("\n # Análise de Distribuição de Oferta:")
        supply_distribution_data = results['supply_distribution_data']
        if supply_distribution_data:
            print(self.analyze_supply_distribution(supply_distribution_data))
        else:
            print("Não foi possível obter dados de distribuição de oferta.")    

        print("\n # Análise EPR (Estimated Price Ratio):")
        epr_data = results['epr_data']
        if epr_data is not None:
            print(self.analyze_epr(epr_data))
        else:
            print("Não foi possível calcular o EPR para análise.")

        print("\n # Análise NVT Ratio (Network Value to Transactions Ratio):")
        nvt_data = results['nvt_data']
        if nvt_data is not None:
            print(self.analyze_nvt_ratio(nvt_data))
        else:
            print("Não foi possível calcular o NVT Ratio para análise.")

        print("\n # Análise da Hash Rate da Rede:")
        hashrate_data = results['hashrate_data']
        if hashrate_data is not None:
            print(self.analyze_network_hashrate(hashrate_data))
        else:
            print("Não foi possível obter dados da hash rate da rede para análise.")

        print("\n # Análise Macroeconômica de Taxas de Juros Globais:")
        global_rates_data = results.get('global_rates_data')
        if global_rates_data is not None:
            print(self.analyze_global_interest_rates(global_rates_data))
        else:
            print("Não foi possível obter dados completos de taxas de juros globais.")
        
        print("\n # Análise de Taxas de Transação:")
        transaction_fees = results.get('transaction_fees')
        if transaction_fees:
            print(self.analyze_transaction_fees(transaction_fees))
        else:
            print("Não foi possível obter dados de taxas de transação.")
        
        print("\n # Análise de Blocos Recentes:")
        recent_blocks = results.get('recent_blocks')
        if recent_blocks:
            print(self.analyze_recent_blocks(recent_blocks))
        else:
            print("Não foi possível obter dados de blocos recentes.")

        print("\n # Análise do Status do Mempool:")
        mempool_status = results.get('mempool_status')
        if mempool_status:
            print(self.analyze_mempool(mempool_status))
        else:
            print("Não foi possível obter dados do status do mempool.")

        print("\n # Análise da Dificuldade de Mineração:")
        mining_difficulty = results.get('mining_difficulty')
        if mining_difficulty:
            print(self.analyze_mining_difficulty(mining_difficulty))
        else:
            print("Não foi possível obter dados de dificuldade de mineração.")

        print("\n # Análise de Dominância do Bitcoin:")
        bitcoin_dominance = results.get('bitcoin_dominance')
        if bitcoin_dominance is not None:
            print(self.analyze_bitcoin_dominance(bitcoin_dominance))
        else:
            print("Não foi possível obter dados de dominância do Bitcoin.")
 
        print("\n \n ## AVALIAÇÃO FINAL:")
        recommendation, score = self.get_master_evaluation(results)
        print(f"|Recomendação: {recommendation}")
        print(f"|Score de confiança: {abs(score):.2f}")
        
        if recommendation == "COMPRAR":
            print("Os indicadores sugerem uma tendência de alta. Considere COMPRAR, mas sempre faça sua própria pesquisa.")
        elif recommendation == "VENDER":
            print("Os indicadores sugerem uma tendência de baixa. Considere VENDER, mas sempre faça sua própria pesquisa.")
        else:
            print("Os indicadores estão mistos. Considere MANTER sua posição atual e monitorar de perto.")
        
        print("\nLembre-se: Esta é uma análise automatizada e não substitui o aconselhamento financeiro profissional.")
        print("Sempre faça sua própria pesquisa e considere sua tolerância ao risco antes de tomar decisões de investimento.")
        
        print("="*50)

    def run(self):
        print("\nBem-vindo ao Analisador de Mercado do Bitcoin Aprimorado :)")
        while True:
            try:
                print("\nO que você gostaria de fazer?")
                print("1. Analisar mercado do Bitcoin")
                print("2. Limpar cache de dados")
                print("3. Sair")
                
                choice = input("\nDigite sua escolha (1-3): ")
                
                if choice == '1':
                    self.analyze()
                elif choice == '2':
                    if os.path.exists(self.cache_file):
                        os.remove(self.cache_file)
                        self.cache = {}
                        print("\nCache de dados limpo com sucesso.")
                        self.logger.info("\nCache de dados limpo pelo usuário.")
                    else:
                        print("Não há cache de dados para limpar.")
                elif choice == '3':
                    print("\nObrigado por usar o Analisador de Mercado do Bitcoin Aprimorado. Até logo!\n")
                    self.logger.info("Sessão de análise encerrada pelo usuário.")
                    break
                else:
                    print("\nEscolha inválida. Por favor, digite 1, 2 ou 3.")
            except Exception as e:
                self.logger.error(f"\nErro inesperado: {str(e)}")
                print("\nOcorreu um erro inesperado. Por favor, tente novamente.")

            print("\n" + "-"*50)

class GoogleTrendsAnalyzer:
    def __init__(self):
        self.keywords = [
            "bitcoin", "crypto", "blockchain", "btc price", "bitcoin trading",
            "bitcoin investment", "cryptocurrency market", "bitcoin news",
            "bitcoin halving", "bitcoin wallet", "bitcoin mining"
        ] 
        self.regions = ["US", "GB", "JP", "KR", "DE", "NG"]  # Principais mercados de criptomoedas
        self.url = "https://trends.google.com/trends/api/dailytrends"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.session = requests.Session()
        self.max_retries = 5
        self.base_delay = 5  # segundos

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
                    logger.warning(f"Rate limit atingido para {region}. Tentativa {attempt+1}/{self.max_retries}. Aguardando {delay:.2f} segundos.")
                    time.sleep(delay)
                else:
                    logger.error(f"Erro ao buscar dados do Google Trends para {region}: {e}")
                    break
        logger.error(f"Falha ao obter dados do Google Trends para {region} após {self.max_retries} tentativas.")
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
            logger.warning("Não foi possível obter scores históricos")
            return 1  # Retornamos 1 para evitar divisão por zero
        
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
            logger.info("\nAnalisando dados do Google Trends...")
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
                logger.warning("Não foi possível obter dados de nenhuma região. Retornando pontuação neutra.")
                return 50  # Pontuação neutra
            
            current_score = total_score / total_volume if total_volume > 0 else 0
            normalized_score = (current_score / historical_average) * 50 if historical_average > 0 else 50
            
            # Garantir que a pontuação esteja no intervalo de 0-100
            final_score = max(0, min(100, normalized_score))
            
            logger.info(f"\nAnálise do Google Trends concluída. Regiões bem-sucedidas: {successful_regions}/{len(self.regions)}")
            return final_score
        
        except Exception as e:
            self.logger.error("\nErro na análise do Google Trends")
            return 50  # Retorna uma pontuação neutra em caso de erro

def main():
    if os.name == 'nt':  # Para Windows
        os.system('title Analisador de Mercado do Bitcoin')
    else:  # Para sistemas Unix/Linux/macOS
        os.system('echo -ne "\033]0;Analisador de Mercado do Bitcoin\007"')

    analyzer = EnhancedBitcoinAnalyzer()
    analyzer.run()
    
if __name__ == "__main__":
    main()

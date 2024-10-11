# Enhanced Bitcoin Market Analyzer (Updated) 
US English version

_If you want a better and more robust version, use the PT-BR (Brazilian Portuguese) version. [Click here to open](https://github.com/FDBnet/Analysis-System-for-Buying-and-Selling-Bitcoin/tree/main/pt-br)._

## Overview 
This project is an enhanced Bitcoin market analysis system that uses price data, technical indicators, funding rates, Google Trends sentiment, on-chain data, and macroeconomic analysis to provide a comprehensive assessment of the market. The system collects data from multiple sources, performs analysis, and provides buy, sell, or hold recommendations based on a weighted rating model.

## Features
- Collects Bitcoin price and volume data via CoinGecko API
- Compares current price to 200-day moving average
- Estimates future funding rates based on historical data
- Gets current funding rates from Binance, Bybit, and OKX exchanges
- Analyzes market sentiment using Google Trends data
- Calculates technical indicators such as RSI and MACD
- Analyzes Bitcoin supply distribution
- Estimates EPR (Estimated Price Ratio), an approximation of SOPR
- Calculates NVT Ratio (Network Value to Transactions Ratio)
- Analyzes Bitcoin network hash rate _[needs correction]_
- Incorporates macroeconomic data (global interest rates) _[needs correction]_
- Weighted rating system considering all indicators
- Makes buy, sell, or hold recommendations based on the final score
- Interactive command-line interface with options for analysis, cache clearing, and output
- Logs activity and errors using Python's logging module

## Requirements 
- Python 3.6+
- Python libraries (see `requirements.txt`)

## Installation 
1. Clone the repository: ``` git clone https://github.com/FDBnet/Analysis-System-for-Buying-and-Selling-Bitcoin.git
   cd Analysis-System-for-Buying-and-Selling-Bitcoin ```
3. Install the dependencies: ``` pip install -r requirements.txt ```

## Usage Run the main script: 
``` python bitcoin_analyzer.py ``` 

The system will start and present an interactive menu. You can choose to perform a new analysis, clear the data cache, or exit the program.

## Contribution 
Contributions are welcome! 
If you would like to contribute, please: 
1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to a branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Disclaimer 
This project is for educational purposes only. It is not financial advice and should not be used as the sole basis for investment decisions. Investing in cryptocurrencies involves significant risks. Always do your own research before investing.

## License 
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
_[Rodrigo S. Magalhães]_

### Threads
*[@rodrigomagalhaes.8](https://www.threads.net/@rodrigomagalhaes.8)*

### Email
*falecom@fortalezadigital.net* 

*digocatu@hotmail.com*

### Project Link:
*[https://github.com/FDBnet/Analysis-System-for-Buying-and-Selling-Bitcoin](https://github.com/FDBnet/Analysis-System-for-Buying-and-Selling-Bitcoin)*


## Help this Project

### *Pix[BR]:*
digocatu@hotmail.com

### *Bitcoin:* 
bc1q63mezfs72jss00xvqhhjzhld33jzm322wn95x3

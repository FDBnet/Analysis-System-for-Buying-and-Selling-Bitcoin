# Sistema de Análise para Compra e Venda de Bitcoin

## Visão Geral

Este projeto é um sistema avançado de análise de Bitcoin que combina análise técnica, on-chain, sentimento de mercado e elementos inspirados nas fórmulas de Midas para fornecer recomendações de trading. O sistema utiliza múltiplas fontes de dados e técnicas de machine learning para oferecer uma análise abrangente do mercado de Bitcoin.

**Nota: Este projeto está em desenvolvimento ativo e busca colaboradores. Se você tem experiência em criptomoedas, análise de dados ou machine learning, sua contribuição seria muito bem-vinda!**

## Funcionalidades

- Análise técnica usando indicadores da biblioteca `ta`
- Análise on-chain com dados da Glassnode (SOPR, NVT, MVRV)
- Análise de sentimento usando Google Trends, CryptoPanic e Twitter
- Previsão de preços usando Random Forest e ARIMA
- Elementos inspirados nas fórmulas de Midas (volatilidade implícita e razão put/call adaptada)
- Sistema de pontuação abrangente
- Recomendações de compra, venda ou manutenção baseadas na pontuação total

## Requisitos

- Python 3.7+
- Bibliotecas Python (veja `requirements.txt`)
- Chaves de API para: Binance, Glassnode, CryptoPanic, Twitter

## Instalação

1. Clone o repositório:
   ```
   git clone https://github.com/FDBnet/Analysis-System-for-Buying-and-Selling-Bitcoin.git
   cd Analysis-System-for-Buying-and-Selling-Bitcoin
   ```

2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

3. Configure as chaves de API:
   Edite o arquivo `config.py` e insira suas chaves de API para Binance, Glassnode, CryptoPanic e Twitter.

## Uso

Execute o script principal:

```
python bitcoin_analyzer.py
```

O sistema iniciará a análise e fornecerá recomendações a cada 10 minutos.

## Contribuição

Estamos ativamente buscando colaboradores para este projeto. Se você tem interesse em contribuir, aqui estão algumas áreas onde precisamos de ajuda:

1. Refinamento dos algoritmos de análise
2. Otimização de performance
3. Adição de novas fontes de dados
4. Melhoria da interface do usuário
5. Testes e validação do sistema

Para contribuir:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Faça commit das suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Aviso Legal

Este projeto é apenas para fins educacionais e de pesquisa. Não é aconselhamento financeiro e não deve ser usado como base única para decisões de investimento. Investir em criptomoedas envolve riscos significativos. Sempre faça sua própria pesquisa e consulte um profissional financeiro antes de investir.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE.md](LICENSE.md) para detalhes.

## Contato

[Rodrigo] - [@rodrigomagalhaes.8]([[https://twitter.com/seu_twitter](https://www.threads.net/@rodrigomagalhaes.8)](https://www.threads.net/@rodrigomagalhaes.8)) - falecom@fortalezadigital.net https://www.threads.net/@rodrigomagalhaes.8

Link do Projeto: [[Link](https://github.com/seu-usuario/bitcoin-analysis-system)]

---

Estou buscando colaboração responsável para fazer este projeto funcionar em sua totalidade. Se você tem experiência em análise de criptomoedas, machine learning, ou desenvolvimento de sistemas de trading, e está interessado em contribuir, por favor, entre em contato. Estou procurando alguém que possa ajudar a refinar os algoritmos, melhorar a precisão das análises e potencialmente expandir o sistema para outras criptomoedas. Toda ajuda é bem-vinda!


## English Version

# Bitcoin Buy and Sell Analysis System

## Overview

This project is an advanced Bitcoin analysis system that combines technical analysis, on-chain analysis, market sentiment, and Midas-inspired elements to provide trading recommendations. The system uses multiple data sources and machine learning techniques to provide comprehensive analysis of the Bitcoin market.

**Note: This project is under active development and is seeking contributors. If you have experience in crypto, data analysis or machine learning, your contribution would be very welcome!**

## Features

- Technical analysis using indicators from the `ta` library
- On-chain analysis with Glassnode data (SOPR, NVT, MVRV)
- Sentiment analysis using Google Trends, CryptoPanic and Twitter
- Price forecasting using Random Forest and ARIMA
- Elements inspired by Midas formulas (implied volatility and adapted put/call ratio)
- Comprehensive scoring system
- Buy, sell or hold recommendations based on the total score

## Requirements

- Python 3.7+
- Python libraries (see `requirements.txt`)
- API keys for: Binance, Glassnode, CryptoPanic, Twitter

## Installation

1. Clone the repository:
```
git clone https://github.com/FDBnet/Analysis-System-for-Buying-and-Selling-Bitcoin.git
cd Analysis-System-for-Buying-and-Selling-Bitcoin
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Configure API keys:
Edit the `config.py` file and enter your API keys for Binance, Glassnode, CryptoPanic, and Twitter.

## Usage

Run the main script:

```
python bitcoin_analyzer.py
```

The system will start analyzing and provide recommendations every 10 minutes.

## Contribution

We are actively seeking contributors to this project. If you are interested in contributing, here are some areas where we need help:

1. Refining the analysis algorithms
2. Optimizing performance
3. Adding new data sources
4. Improving the user interface
5. Testing and validating the system

To contribute:

1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Disclaimer

This project is for educational and research purposes only. It is not financial advice and should not be used as the sole basis for investment decisions. Investing in cryptocurrencies involves significant risks. Always do your own research and consult a financial professional before investing.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

[Rodrigo] - [@rodrigomagalhaes.8]([https://www.threads.net/@rodrigomagalhaes.8](https://www.threads.net/@rodrigomagalhaes.8)) - falecom@fortalezadigital.net

Project Link: [[Link](https://github.com/FDBnet/Analysis-System-for-Buying-and-Selling-Bitcoin)]

---
I am looking for responsible collaboration to make this project work in its entirety. If you have experience in cryptocurrency analysis, machine learning, or trading system development and are interested in contributing, please get in touch. I'm looking for someone who can help refine the algorithms, improve the accuracy of the analysis, and potentially expand the system to other cryptocurrencies. Any help is welcome!

# Sistema de Análise de Mercado do Bitcoin (Versão Legada)

## Visão Geral

Este projeto é um sistema avançado de análise de mercado do Bitcoin que combina análise técnica, dados on-chain, sentimento de mercado e modelos de machine learning para fornecer recomendações de trading. O sistema utiliza múltiplas fontes de dados e técnicas para oferecer uma análise abrangente do mercado de Bitcoin.

## Funcionalidades

- Coleta de dados de preço, volume e capitalização de mercado do Bitcoin através da CoinGecko API e APIs alternativas
- Análise técnica usando diversos indicadores da biblioteca `ta` 
- Incorporação de dados de sentimento do Google Trends
- Análise de atividade e engajamento com Bitcoin no GitHub
- Previsão de preços usando modelos Random Forest e ARIMA
- Cálculo de pontuação de risco baseada no Value-at-Risk e Max Drawdown históricos
- Sistema de pontuação abrangente considerando análise técnica, dados de mercado, sentimento, previsões e risco
- Recomendações de compra, venda ou manutenção baseadas na pontuação total
- Interface de linha de comando interativa

## Requisitos

- Python 3.6+
- Bibliotecas Python (veja `requirements.txt`)

## Instalação

1. Clone o repositório:
   ```
   git clone https://github.com/FDBnet/Analysis-System-for-Buying-and-Selling-Bitcoin/tree/main/legado.git
   cd Analysis-System-for-Buying-and-Selling-Bitcoin
   ```

2. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

## Uso

Execute o script principal:

```
python bitcoin_analyzer.py
```

O sistema iniciará e apresentará um menu interativo. Você pode escolher realizar uma nova análise, ver os resultados da última análise ou sair do programa.

## Contribuição

Contribuições são bem-vindas! Se você quiser contribuir, por favor:

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Faça commit das suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`) 
5. Abra um Pull Request

## Aviso Legal

Este projeto é apenas para fins educacionais. Não é aconselhamento financeiro e não deve ser usado como base única para decisões de investimento. Investir em criptomoedas envolve riscos significativos. Sempre faça sua própria pesquisa antes de investir.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Contato

[Rodrigo] - [@rodrigomagalhaes.8](https://www.threads.net/@rodrigomagalhaes.8) - falecom@fortalezadigital.net

Link do Projeto: [https://github.com/FDBnet/Analysis-System-for-Buying-and-Selling-Bitcoin](https://github.com/FDBnet/Analysis-System-for-Buying-and-Selling-Bitcoin)

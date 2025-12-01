# Quantitative-Strategy-Research

## About This Project

In this application, I explore several different novel trading strategies, as well as create the technical infrastructure to assess, backtest, and extend said strategies within different contexts.

### Embedding Strategies

These novel trading strategies make use of textual embedding representations of companies per year. These textual embeddings were developed by myself in a separate, ongoing financial research project overseen by my supervising professor. These embeddings represent aspects of companies including business descriptions, ongoing internal projects, public sentiments, sector involvements, and assessments of technology, labour and capital. The sources of this textual data range across company 10K reports, patent filings, job postings, and media coverage; this has been accumulated and applied to several-hundred-thousand company-years. As the accumulated source text is licensed data, I am not permitted to publicly share the text nor the corresponding embeddings in my Pinecone Database. Note that the strategies outlined in this repository are not specific to these embeddings and can be applied to any meaningful company embeddings.

Results indicate strong performance for several of the embedding-based strategies when backtested over various time frames -- more complete logs are available in the Outputs folder. Despite these optimistic results, the strategies have not been assessed to a professional standard for any meaningful level of risk quantification or expected value. Additionally, there is some marginal time-leakage of future data: firstly, the sample of candidate companies which the strategies select from suffers from some survivorship bias, as the dataset of company embeddings is skewed towards companies for which data was available over a longer time period (possibly in the future). Secondly, the language-model used to compute embedding vectors for textual bodies were not enforced to be strictly trained on data prior to the time of trading decisions -- I am currently working on developing my own embedding models using contrastive learning that will enforce strict temporal integrity.

### Additional Features

Furthermore, I have designed and created a simplified backtesting simulation tool that allows users to create their own trading strategies and evaluate their performances on historical data over different time windows. My motivation for creating this was partially to compare how commonly preached naive investing strategies actually stack up against each other, and also partially to have a sandbox in which I can quickly write-up and test different strategies. Although this system lacks the rigour of professional-standard backtesting libraries such as bt, it is very user friendly for curious traders and allows for quick and significant intuition to be built for potentially complex user strategies.

Several simple example strategies are included in this repository to showcase potential approaches and formatting. Of course, there is plenty of room for creativity to bring all kinds of techniques into the picture using the available parameters. Note that there are various limitations of this system in regards API quality (Yahoo Finance free version) and analytical tool availability.

### How to Use

To assess your own strategy, emulate the examples shown in the repository: design the trading strategy to your liking, and then initialize the backtesting simulation with command 'python -m SimpleBacktester.ResultSimulator'.

To recreate the results of the embedding trading strategies shown: you must first provide your own text embedding representations of the companies, and then run strategies with the command 'python EmbeddingStrategy{x}.py'.

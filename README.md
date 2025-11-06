# Quantitative-Trading

## About This Project

In this application, I explore several different novel trading strategies as well as create the technical infrastructure to assess, backtest, and extend said strategies within different contexts.

### Embedding Strategies

These novel trading strategies make use of textual embedding representations of companies per year. These textual embeddings were developed by myself in a separate, ongoing financial research project overseen by my supervising professor. These embedding represent aspects of companies including business descriptions, ongoing internal projects, public sentiments, sector involvements, and assessments of technology, labour and capital. The sources of this textual data range across company 10K reports, patent filings, job postings, and media coverage; this has been accumulated and applied to several-hundred-thousand company-years. To access these embeddings, please contact myself for access to the Pinecone dataset storing said vectors. Note that the strategies outlined in this repository are not specific to these embeddings and can be applied to any form of company embedding representations of interest.

Results show strong performance for several embedding-based strategies when backtested for various time frames -- complete logs are available in the Outputs folder. Despite these optimistic results, the strategies have not been assessed to a professional standard for any meaningful level of risk quantification or expected value. Additionally, there is some marginal time-leakage of future data: firstly, the sample of candidate companies from which the strategies draw suffer from some survivorship bias as our dataset of company embeddings is skewed towards companies for which data was available over a reasonable time period. Secondly, the model used to compute embedding vectors for textual bodies were not enforced to be strictly trained on data prior to the time of trading decisions -- I am currently working on developing my own embedding models using contrastive learning that will enforce strict temporal integrity.

### Other Infrastructure

Additionally, I have designed and created a simplified backtesting simulation that allows users to create their own trading strategies and evaluate performance on historical data over different time windows and stocks. My motivation for creating this was partially to see how commonly preached naive strategies actually stack up against each other, and also partially to have a sandbox in which I quickly write-up and test different basic strategies. Although this backtesting lacks the rigour of professional standard backtesting systems such as bt, it is very user friendly for curious beginners and allows for quick and significant intuition to be built around potentially complex user strategies.

Several simple example strategies are included in this repository to showcase potential structuring for someone curious to create their own trading strategies. Of course, there is plenty of room for creativity to bring all kinds of techniques into the picture within the lines of the available parameters.

Note that there are various limitations with this system if attempting to utilize as a professional tool -- both along the lines of API quality and analytical tool availability.

### How to Use

To assess your own strategy: emulate the examples shown in the repository, design the trading strategy to your liking, and run the backtest assessment of interest with command 'python -m SimpleBacktester.ResultSimulator'.

To verify results of the embedding trading strategies shown, please request access from myself for the needed Pinecone API Key, and then design and run strategies from the user entry point of command 'python EmbeddingStrategy{x}.py'.

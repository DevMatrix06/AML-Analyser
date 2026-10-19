# AML Analyser: catching dirty money by reading the shape of the network

Banks usually look at transactions one at a time. Money laundering rarely shows up that way. It shows up in the
*structure*: money split across many accounts, passed along in chains, and collected again somewhere else. A flat table of
transactions hides that structure, so this project models the transactions as a **graph** and trains
**Graph Neural Networks (GraphSAGE)** to score how suspicious each transaction is. A **Streamlit dashboard** lets you
explore the results, from a global overview down to the neighbourhood around a single account.

It covers two different datasets and two different tasks, side by side:

| | **AMLSim** (banking) | **Elliptic** (Bitcoin) |
|---|---|---|
| Data | Synthetic bank transfers: 1,323,234 transactions between 9,999 accounts, of which 1,719 (0.13%) are fraud | A real Bitcoin transaction graph: 203,769 transactions and 234,355 edges. 4,545 are labelled illicit, 42,019 licit and 157,205 unlabelled |
| Task | **Edge classification**: give every transfer a suspicion score from 0 to 1 | **Node classification**: give every transaction a probability of being illicit |
| Model | `AMLGraphSAGE`: two GraphSAGE layers (64 then 32 dimensions); the sender and receiver embeddings are joined and passed to a linear layer | `BitcoinGraphSAGE`: two GraphSAGE layers and a two-class linear layer with softmax |
| Scores saved to | `outputs/suspicion_scores.csv` | `outputs/bitcoin_predictions.csv` |

## How training works
- **Heavy class imbalance** is handled with a weighted binary cross-entropy loss (the fraud class is weighted by the ratio of normal to fraudulent examples), so missing a fraud case costs much more than a false alarm.
- Adam optimiser (learning rate 0.01), dropout of 0.5, 100 epochs. The best checkpoint is kept: by validation ROC-AUC for AMLSim and by validation F1 for Elliptic.
- Trained models are saved in `outputs/` (`model_checkpoint.pth`, `model_checkpoint_bitcoin.pth`).

## The dashboard
Pick a project in the sidebar, then a view:

- **Global Dashboard**
  - AMLSim: transactions grouped by suspicion score, a top-10 leaderboard of high-risk accounts, an interactive network map of the most suspicious flows, and a filterable transaction table.
  - Elliptic: the distribution of predicted illicit scores, average risk over time steps, the top predicted high-risk transactions among the unlabelled ones, and an interactive network map.
- **Account / Address Deep Dive**: investigate one account (or one Bitcoin transaction) and see the network around it.

## Run it
```bash
git clone https://github.com/DevMatrix06/AML-Analyser.git
cd AML-Analyser/AML-Transaction-Graph-Analyser
pip install -r requirements.txt
streamlit run streamlit_app.py
```
The repository already contains the processed AMLSim graph, the trained checkpoints and the saved scores, so the dashboard
works straight after cloning. Run the commands from inside `AML-Transaction-Graph-Analyser`, because the paths are relative.

To retrain:
```bash
# AMLSim (edge classification)
python src/train.py
python src/predict.py

# Elliptic (node classification)
# first download elliptic_txs_features.csv from the public Elliptic data set into data/raw/ (the file is too large for Git)
python src/data_preprocessing_bitcoin.py
python src/train_bitcoin.py
python src/predict_bitcoin.py
```

## Results, honestly
- **AMLSim:** the best validation ROC-AUC recorded for the model is **0.5649**, on data where only 0.13% of transactions are fraud. That is only slightly better than random. The base model learns from the graph structure alone: it uses a placeholder node feature, and the GraphSAGE layers do not take the edge attributes (amount, type) as input. It is a working end-to-end pipeline and a baseline, not a finished detector.
- **Elliptic:** the test F1 score recorded for the model is **0.85** (0.8497, used by the dashboard), measured on the labelled Elliptic transactions. `python src/train_bitcoin.py` prints the score when you retrain. There is no non-graph baseline in this repository yet, so it does not show how much the graph itself adds.

## Ideas to improve it
- Feed edge features (amount, time) into the model, for example with an edge-aware convolution, so it can use more than the topology.
- Pick the decision threshold on the validation set instead of using a fixed 0.5.
- Use a time-based train/test split for Elliptic, and compare against a non-graph baseline such as gradient boosting on the node features, to show what the graph really adds.

## Project layout
```
AML-Transaction-Graph-Analyser/
  streamlit_app.py        dashboard entry point
  ui/                     dashboard pages for AMLSim and Elliptic
  src/                    models, training and prediction scripts
  data/raw, data/processed  datasets and the processed graph
  outputs/                scores, checkpoints and generated network maps
```

Built with Python, PyTorch, PyTorch Geometric, scikit-learn, NetworkX, PyVis and Streamlit.

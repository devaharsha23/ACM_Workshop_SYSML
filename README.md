# ACM Workshop on Systems for ML (ACM_WS_SYSML)


- $c[i]$ = Set of labels of classes that client $i$ trains its model on. 
- $n_c$ = number of all classes
- $d_i$ = $len(classes[i])$ . (Max value = $n_c$ = 10)
 - $l_i$ = appeared in the past window how many times?
	 - This won't allow a client to be picked in a row, more than a certain set number of times (would be heuristic based, say `3` in our case)
 - $acc_{avg}[i]$ - The average increase/decrease in Global accuracy that was contributed by a selection of a group of clients in which client $i$ was a part of.

## Goal


Every iteration, we have to choose $i, j, k$ so that we can maximise the function below (We just have to take a monotonically increasing function of the parameters used below)
	(To include : inverse of entropy also after the last bracket)

$$
max_{{i, j, k}}{len(c_{i} \cup c_{j} \cup c_{k})}*(acc_{avg}[i] * acc_{avg}[j] * acc_{avg}[k])
$$
Could be + instead of * as well. (Not sure, would need some time to think)

Subject to constraints:
$$
l_{i} < 3$$
$$l_{j} < 3 
$$
$$
l_{k} < 3
$$

(Here, 3 is a heuristic. If we have clients that contribute a good global accuracy and have diverse enough data, we could set this to 2 as well)

(Note). : Explain that if repeated selection occurs for one of the client, it may increase accuracy but will stagnate at a point and won't be able to add much benefit to the global accuracy, slowly it's avg accuracy will go down, causing the algorithm to ultimately converge and end up choosing the client which was starved (os related).



A federated learning implementation for energy-efficient distributed training.

## Repository Structure

```
ACM_WS_SYSML/
├── configs/              # Configuration files
│   └── config.yaml      # Main configuration file
├── models/              # Model implementations
│   └── CNN.py          # CNN model implementation
├── initial_models/      # Pre-trained model checkpoints
├── client.py           # Client implementation
├── server.py           # Server implementation
├── run_exp.py          # Main experiment runner
└── requirements.txt    # Project dependencies
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download data
```bash
wget "https://www.dropbox.com/scl/fi/3bqhiz8uzcol2ge28ruce/data.zip?rlkey=ljhilzs8qam2m0mohru3otmx7&st=wkc7ehq7&dl=0" -O data.zip
unzip data.zip  
```

4. Run an experiment:
```bash
python run_exp.py configs/config.yaml
```

## Configuration

The `config.yaml` file contains experiment parameters including:
- Model architecture settings
- Training parameters
- Client/server configuration
- Dataset settings

## Models

Currently supported models:
- CNN (Convolutional Neural Network)

## License

This project is licensed under the MIT License.

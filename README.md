# Word-level text generator with LSTM

This example trains a multi-layer RNN-LSTM on a language modeling task.
The training script uses the PTB dataset, provided.
The trained model can then be used by the generate script to generate new text.

## Dependencies

Python

```
$ sudo apt-get install python3 python3-pip
```

Pytorch. Make sure you have also CUDA available in you machine.

```
$ conda install pytorch
```

NumPy

```
$ pip install numpy
# $ conda install numpy
```

Matplotlib

```
$ pip install matplotlib
# $ conda install matplotlib
```

## Training and Testing

You can train and test the model with `main.py` script. By default it uses PTB dataset from the `data` folder.
This script first train the model with train data set, then test the model on test dataset.
The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --embsize EMSIZE    size of word embeddings
  --nhid NHID        number of hidden units per layer
  --nlayers NLAYERS  number of layers
  --lr LR            initial learning rate
  --epochs EPOCHS    upper epoch limit
  --batch-size N     batch size
  --bptt BPTT        sequence length
  --dropout DROPOUT  dropout applied to layers (0 = no dropout)
```

With these arguments, a variety of models can be tested.
As an example, the following arguments produce better models:

```bash
python main.py --emsize 650 --nhid 650 --bptt 35 --dropout 0.5     # Test perplexity of 82.77
python main.py --emsize 650 --nhid 650 --bptt 60 --dropout 0.5     # Test perplexity of 84.61
python main.py --emsize 1500 --nhid 1500 --bptt 35 --dropout 0.65  # Test perplexity of 79.25
python main.py --emsize 1500 --nhid 1500 --bptt 60 --dropout 0.65  # Test perplexity of 82.64
```

## Generate text

You can generate a text using the pre-trained model with `generate.py` script. Remember, the model should be trained first before generating.
The `generate.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help                    show this help message and exit
  --words WORDS                 number of words to generate
  --temperature TEMPERATURE     temperature - higher will increase diversity
```

With these arguments, a text can be generated.

```bash
python generate.py --words 2000
```

For any questions, do not hesitate to contact me or open an issue! :point_down::point_down::point_down::point_down::point_down::mega:  
zihadul.azam@studenti.unitn.it

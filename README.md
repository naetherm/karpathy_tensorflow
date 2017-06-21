# karpathy_tensorflow
Simple implementation of the kaparthy code (http://karpathy.github.io/2015/05/21/rnn-effectiveness/) in Tensorflow.

This implementation was mainly done for learning purposes and further deepening my understanding of recurrent neura networks with tensorflow.


## Installing Tensorflow

For installing Tensorflow just follow the instruction of the official homepage [1]. For this implementation the latest release of Tensorflow was used (version 1.2.0).



[1] https://www.tensorflow.org/install/

## Usage

Start training:

```bash
python train.py --datapath=shakespeare/
```

The default model to use is ```bash rnn``` but can be changed by the command line parameters.

```bash
python train.py --data_path=shakespeare/ --model=lstm
```

Available models are: ```bash rnn, lstm, gru and nas```.

After training a model you can run a sample on it:

```bash
python sample.py --data_path=shakespeare/
```

The length of the sample that should be created can be changed by the parameter ```bash --n```:

```bash
python sample.py --data_path=shakespeare/ --n=1000
```

## Further samples

The wikipedia ```bash train.txt``` can be get [here](https://mega.nz/#!Qa4z0DiL!BBNviuoNUN-awop3inFs_MyilijwLSy6O36-feGUMr8), and the linux ```bash train.txt``` can be get [here](https://mega.nz/#!9S5xhaIY!4_grlN6D3yPJYBCfiT_uR0N-tXkFlmHXWuafyL7KwaQ).
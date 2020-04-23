## Instructions for TorchText on Longhorn

To run `pytorch_wikitext_rnn.py`, you need TorchText 0.5 which has sentencepiece as a requirement. Sentencepiece must be built from source on Longhorn.

```
$ git clone https://github.com/google/sentencepiece.git
$ cd /path/to/sentencepiece
$ mkdir build
$ cd build
$ cmake -DCMAKE_PREFIX_PATH=/path/to/install ..
$ make -j $(nproc)
$ make install
$ export LD_LIBRARY_PATH=/path/to/install/lib:$LD_LIBRARY_PATH  # can add to ~/.bashrc
$ cd /path/to/sentencepiece/python
$ python setup.py build
$ python setup.py install
$ pip install torchtext==0.5
```



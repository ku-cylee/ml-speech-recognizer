# Speech Recognizer using HMM Model and Baum-Welch Algorithm

## Summary

This program is a speech recognizer program which is implemented using Hidden Markov Model and Baum-Welch Algorithm. The project was created for COSE362 Machine Learning Course. The program is **incomplete**. 

## Contributor

This project was contributed by Lee Chanyoung([@ku-cylee](https://github.com/ku-cylee/))

## How to run

This project requires **Python version newer than 3.6.** Locations of all terminal scripts below are project root directory.

### Create Virtual Environment

Create virtual environment and activate it. For Windows,
```sh
$ pip -m venv .venv
$ .venv\Scripts\activate
```

For macOS and Linux OS,
```sh
$ pip3 -m venv .venv
$ source .venv/bin/activate
```

### Install Required Libraries

Install the libraries required.
```sh
$ pip install -r requirements.txt
```

### Config .env File

Now, create ".env" file at the project directory, and specify your file directories.

* `HMM_PATH`: HMM model file path
* `TRANSCRIPT_PATH`: Transcripts file path which contains monophones
* `TRAIN_DATA_PATH`: Path of directory which contains the whole file with vector values for each transcripts

```plain
HMM_PATH=<hmm-model-path>
TRANSCRIPT_PATH=<transcript-path>
TRAIN_DATA_PATH=<train-data-dir-path>
```

For example,
```plain
HMM_PATH=~/data/hmm.txt
TRANSCRIPT_PATH=~/data/trn_mono.txt
TRAIN_DATA_PATH=~/data/trn
```

Location of each training data file should be able to be found by the file path specified inside the transcripts file INSIDE the `TRAIN_DATA_PATH`. For example, if your `TRAIN_DATA_PATH` is ~/data/trn, and your transcripts file contains trn/f/ac/111111.lab as vector values' file path, its actual path should be ~/data/trn/trn/f/ac/111111.txt.

### Run the program

Now, you can run the program:
```sh
$ python ./src/main.py
```

## Limitations

This project is incomplete. It is able to accumulate values for each transcripts, but is not able to calculate change values of parameters for each iteration. 

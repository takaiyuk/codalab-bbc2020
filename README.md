# Basketball Behavior Challenge: BBC2020

https://competitions.codalab.org/competitions/23905

## Setup & RUN

### Data

Download files from [here](https://competitions.codalab.org/competitions/23905#participate-get_starting_kit) and locate them in data/raw

```
$ ls ~/work/codalab/codalab-bbc2020/data/raw
test  train
```

### RUN
```
$ ./initialize.sh
$ ./docker/pull.sh && ./docker/run.sh codalab-bbc2020 && ./docker/exec.sh codalab-bbc2020
root@xxxxxx:/workspace# . /venv/bin/activate
(venv) root@xxxxxx:/workspace# python run.py --fe fe000 --run run000
```

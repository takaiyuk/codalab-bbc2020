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
(venv) root@xxxxxx:/workspace# python run.py --fe fe006 --run run009
```

- blending
```
(venv) root@xxxxxx:/workspace# python run.py --blend blend000
```

## Fun Things

### Visualization

Using [DrawNBACourt](https://github.com/takaiyuk/codalab-bbc2020/blob/master/src/utils/visualize.py#L20), which is insipired by [BasketballAnalyzeR](https://github.com/sndmrc/BasketballAnalyzeR/blob/master/R/drawNBAcourt.R), and matplotlib animation, we can get gif images of each play.

- positive example

![negative example](https://raw.githubusercontent.com/takaiyuk/codalab-bbc2020/master/notebooks/gif/0000.gif)

- negative example

![negative example](https://raw.githubusercontent.com/takaiyuk/codalab-bbc2020/master/notebooks/gif/0401.gif)

## Ref.

- [Basketball Behavior Challenge BBC2020 で4チーム中2位に - Tak's Notebook](https://takaishikawa42.hatenablog.com/entry/2020/09/03/234551)

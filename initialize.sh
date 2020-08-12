#!/bin/bash

function checkdir () {
    if [ $(ls -1 | wc -l) -ne 1 ]; then
        exit 1
    fi
}

function makedirs () {
    mkdir -p data/external
    mkdir -p data/interim
    mkdir -p data/processed
    mkdir -p data/raw

    mkdir -p docker

    mkdir -p docs

    mkdir -p models/importance
    mkdir -p models/model
    mkdir -p models/optuna
    mkdir -p models/prediction
    mkdir -p models/pretrained

    mkdir -p notebooks

    mkdir -p references

    mkdir -p shell

    mkdir -p src/config
    mkdir -p src/data
    mkdir -p src/features
    mkdir -p src/models
    mkdir -p src/utils

    mkdir -p submissions
}

function touch_keep () {
    touch data/external/.gitkeep
    touch data/interim/.gitkeep
    touch data/processed/.gitkeep
    touch data/raw/.gitkeep

    touch docker/.gitkeep

    touch docs/.gitkeep

    touch models/importance/.gitkeep
    touch models/model/.gitkeep
    touch models/optuna/.gitkeep
    touch models/prediction/.gitkeep
    touch models/pretrained/.gitkeep

    touch notebooks/.gitkeep

    touch references/.gitkeep

    touch shell/.gitkeep

    touch src/config/.gitkeep
    touch src/data/.gitkeep
    touch src/features/.gitkeep
    touch src/models/.gitkeep
    touch src/utils/.gitkeep

    touch submissions/.gitkeep
}

function touch_init () {
    touch src/__init__.py
    touch src/config/__init__.py
    touch src/data/__init__.py
    touch src/features/__init__.py
    touch src/models/__init__.py
    touch src/utils/__init__.py

    touch docker/pull.sh
    touch docker/run.sh
    touch docker/exec.sh
    touch docs/competition.md
    touch src/config/fe000.yml
    touch src/config/run000.yml
    touch shell/download.sh
    touch shell/submit.sh
    touch src/const.py
    # touch .gitignore
    touch README.md
    touch run.py
}

function chmod_shell () {
    chmod +x ./docker/exec.sh
    chmod +x ./docker/pull.sh
    chmod +x ./docker/run.sh

    chmod +x ./run.sh

    chmod +x ./shell/download.sh
    chmod +x ./shell/submit.sh
}

function git_init () {
    git init
}

checkdir
makedirs
touch_keep
touch_init
chmod_shell
# git_init

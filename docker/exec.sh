#!/bin/bash
sudo docker start $1 && sudo docker exec -it $1 /bin/bash

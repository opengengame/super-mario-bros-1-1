## Setup

To set up a Python virtual environment with the required dependencies, run:
```shell
# create virtual environment
python3 -m venv ./envs
source ./envs/bin/activate
# update pip, setuptools and wheel
pip3 install --upgrade pip setuptools wheel
# install all required packages
pip3 install -r requirements.txt
```

To export the install requirements, run:
```shell
pip3 freeze > requirements.txt
```

Once done with virtual environment, deactivate with command:
```shell
deactivate
```
then delete venv with command:
```shell
rm -r ./envs
```
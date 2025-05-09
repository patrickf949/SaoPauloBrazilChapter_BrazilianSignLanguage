# Web API

### Technologies used

- Python3

### Steps to test Locally

- Check into the backend directory

```bash
cd webapp/backend
```

- Create a virtual environment and install the packages required

```bash
python -m venv venv # to create the virtual environment
source venv/bin/activate # to activate the virtual environment
pip install -r requirements.txt # to install the necessary packages
```

- Run the Application

```bash
uvicorn main:app --port 5000 --reload
```



# Scorecard Project

Welcome to the Scorecard project! 

This project allows you to scan SEDRA scorecards process rider data and return back editable results from your form.

## Getting Started

To get started with the Scorecard project, follow these steps:

### Prerequisites

Make sure you have the following prerequisites installed on your machine:

- Python 3.10.11 [Download](https://www.python.org/downloads/release/python-31011/)

### Setting up a Virtual Environment (Recommended but Optional)
#### Warning
It's always a good practice to use a virtual environment to manage project dependencies.

If you do not use a virtual environment then any conflicting global dependencies will be updated.

This can potentially break other projects that rely on an older version of the replaced dependency. 

#### Configuration
Follow these steps to set up a virtual environment:

```bash
# Install virtualenv package
pip install virtualenv
# Create a virtual environment
python -m virtualenv [path]/[environment name]

# Activate the virtual environment
# On Windows 10
[path]\[environment name]\Scripts\activate
# On Windows 11
[path]\[environment name]\bin\activate

# This will enter your current terminal into the newly created virtual environment
```

### Pulling from the GitHub repository

#### Using Virtual Enviroment

I recommend storing the project in the same parent folder of your virtual environment. This is because each new instance of the terminal will need to reactive the environment before running the server

#

### Pull

Once in the desired directory git clone this repo using the command below to create a folder with all of the project source code:

```bash
git clone https://github.com/Zackary-Crow/scorecard.git
# Navigate to repo folder
cd scorecard
```

Next, we will install the required packages, to do so execute:

`pip install -r requirements.txt`

This will install all of the required packages into the environment

Now run `python manage.py runserver`

You should see some terminal output start to appear and the server will be active when you see the lines
```bash
Django version 4.2.7, using settings 'backend.settings'
Starting ASGI/Daphne version 4.0.0 development server at http://127.0.0.1:8000/
Quit the server with CTRL-BREAK.
```

Using the listed IP and port you can connect to the webpage and interact with the page


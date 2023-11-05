python3.9 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

python manage.py makemigrations --noinput
python manage.py migrate --noinput

python manage.py collectstatic --noinput --clear
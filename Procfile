release: python manage.py collectstatic --noinput
web: python -m gunicorn config.wsgi:application --bind 0.0.0.0:${PORT}

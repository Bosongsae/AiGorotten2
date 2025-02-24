#!/bin/bash
set -e
python3 -m pip install --upgrade pip
python3 -m pip install -e backend
python3 backend/main.py
python3 -m gunicorn fastapi_app:app -c backend/main.py

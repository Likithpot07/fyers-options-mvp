FYERS Options Predictor

Real-time NIFTY options monitoring and ML prediction using FYERS API V3.

Setup

1. Create a FYERS Account

Create and activate a FYERS trading account.

2. Create a FYERS API App

Create an app in the FYERS API dashboard.

Use this Redirect URI:

http://127.0.0.1:8000/callback

Copy:

Client ID

Secret Key

3. Create .env

Create a .env file in the project root:

FYERS_CLIENT_ID=YOUR_CLIENT_ID
FYERS_SECRET_KEY=YOUR_SECRET_KEY
FYERS_ACCESS_TOKEN=

Do not commit .env to GitHub.

4. Generate Access Token

Run:

python auth.py

Then:

Open the generated FYERS login URL.

Log in and authorize the app.

Copy the auth_code from the redirected URL.

Paste it into the terminal.

The script generates the access_token and saves it to .env.

5. Install Dependencies

Create and activate a Python 3.9 virtual environment, then run:

pip install -r requirements.txt
pip install fyers-apiv3==3.1.16 --no-deps

6. Run the Pipeline

python 01_fetch_data.py
python 02_build_dataset.py
python 03_train_model.py
python 04_backtest.py
python 05_diagnostics.py
python 06_threshold_tuning.py
python 07_setup_score_tuning.py
python 08_walk_forward.py

7. Run Live Predictor

python live_data.py

Do Not Commit

.env
venv/
*.log
from pathlib import Path
from fyers_apiv3 import fyersModel

from config import CLIENT_ID, SECRET_KEY

REDIRECT_URI = "http://127.0.0.1:8000/callback"

session = fyersModel.SessionModel(
    client_id=CLIENT_ID,
    secret_key=SECRET_KEY,
    redirect_uri=REDIRECT_URI,
    response_type="code",
    grant_type="authorization_code"
)

login_url = session.generate_authcode()

print("Open this URL:")
print(login_url)

auth_code = input("\nPaste auth_code here: ").strip()

session.set_token(auth_code)
response = session.generate_token()

if response.get("s") != "ok":
    print("ERROR:", response)
    raise SystemExit

access_token = response["access_token"]
refresh_token = response.get("refresh_token")

env_path = Path(".env")
lines = env_path.read_text().splitlines()

updated_lines = []

for line in lines:
    if line.startswith("FYERS_ACCESS_TOKEN="):
        updated_lines.append(f"FYERS_ACCESS_TOKEN={access_token}")
    elif line.startswith("FYERS_REFRESH_TOKEN="):
        updated_lines.append(f"FYERS_REFRESH_TOKEN={refresh_token}")
    else:
        updated_lines.append(line)

if not any(line.startswith("FYERS_REFRESH_TOKEN=") for line in lines):
    updated_lines.append(f"FYERS_REFRESH_TOKEN={refresh_token}")

env_path.write_text("\n".join(updated_lines) + "\n")

print("\nAccess token refreshed and saved to .env")
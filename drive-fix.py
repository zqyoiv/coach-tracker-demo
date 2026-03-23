from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = "key.json"
ROOT_FOLDER_ID = "1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkmlCIK"

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
service = build("drive", "v3", credentials=creds)

def list_folder(folder_id, indent=0):
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(
        q=query,
        fields="files(id, name, mimeType)",
        includeItemsFromAllDrives=True,
        supportsAllDrives=True
    ).execute()

    items = results.get("files", [])
    for item in items:
        print("  " * indent + f"- {item['name']} ({item['mimeType']})")
        if item["mimeType"] == "application/vnd.google-apps.folder":
            list_folder(item["id"], indent + 1)

list_folder(ROOT_FOLDER_ID)
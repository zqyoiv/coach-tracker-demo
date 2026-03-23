import google.auth
from googleapiclient.discovery import build

credentials, project = google.auth.default(
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

service = build('drive', 'v3', credentials=credentials)

folder_id = '1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkmlCIK' 
results = service.files().list(
    q=f"'{folder_id}' in parents and trashed = false",
    fields="files(id, name)").execute()

items = results.get('files', [])
print(f"found {len(items)} files：")
for item in items:
    print(f"- {item['name']}")
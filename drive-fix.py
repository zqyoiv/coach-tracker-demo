from google.oauth2 import service_account
from googleapiclient.discovery import build

# 认证
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'key.json'

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)

service = build('drive', 'v3', credentials=creds)

# 你的 folder ID
folder_id = '1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkmlCIK'

# 查询文件
query = f"'{folder_id}' in parents"

results = service.files().list(
    q=query,
    fields="files(id, name)"
).execute()

files = results.get('files', [])

# 👇 就加在这里
print("Files found:", files)

# 可选：更清晰一点
if not files:
    print("⚠️ No files found in this folder.")
else:
    for file in files:
        print(f"{file['name']} ({file['id']})")
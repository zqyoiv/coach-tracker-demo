import google.auth
from googleapiclient.discovery import build

credentials, _ = google.auth.default(
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

service = build('drive', 'v3', credentials=credentials)

# 2. 测试列出文件夹（替换为你截图中那个 long ID）
folder_id = '1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkm1CIK' 
results = service.files().list(
    q=f"'{folder_id}' in parents and trashed = false",
    fields="files(id, name)").execute()

items = results.get('files', [])
print(f"found {len(items)} files：")
for item in items:
    print(f"- {item['name']}")
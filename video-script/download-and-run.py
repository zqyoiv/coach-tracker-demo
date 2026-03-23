import google.auth
from googleapiclient.discovery import build

credentials, project = google.auth.default(
    scopes=['https://www.googleapis.com/auth/drive.readonly']
)

if not credentials:
    print("错误：未能获取到 VM 身份凭证！")

service = build('drive', 'v3', credentials=credentials)

folder_id = '1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkm1CIK'
try:
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        fields="files(id, name)").execute()
    print(f"成功！找到 {len(results.get('files', []))} 个文件。")
except Exception as e:
    print(f"调用失败，错误详情: {e}")
import google.auth
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# 强制指定最高权限 Scope
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# 1. 获取凭证
credentials, project = google.auth.default()

# 2. 核心操作：如果凭证没有 scopes，或者 scopes 不对，强制刷新它
if credentials.requires_scopes:
    credentials = credentials.with_scopes(SCOPES)

# 3. 确保凭证是有效的（刷新一下）
credentials.refresh(Request())

# 4. 构造 service
service = build('drive', 'v3', credentials=credentials)

folder_id = '1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkm1CIK'
try:
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        fields="files(id, name)").execute()
    print(f"成功！找到 {len(results.get('files', []))} 个文件。")
except Exception as e:
    print(f"调用失败，错误详情: {e}")
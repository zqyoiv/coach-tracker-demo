import google.auth
# 引入专门针对虚拟机的库
from google.auth import compute_engine
from googleapiclient.discovery import build

# 1. 强制直接去虚拟机的“心跳”里拿权限，跳过所有本地 JSON 文件
credentials = compute_engine.Credentials(
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

# 2. 构造服务
service = build('drive', 'v3', credentials=credentials)

folder_id = '1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkm1CIK'

try:
    print("正在直接使用虚拟机身份访问 Drive...")
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        fields="files(id, name)").execute()
    
    items = results.get('files', [])
    print(f"🎉 成功了！找到 {len(items)} 个文件。")
except Exception as e:
    print(f"❌ 还是不行。错误详情: {e}")
import google.auth
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

def run_test():
    # 1. 显式获取凭证
    credentials, project = google.auth.default()
    
    # 2. 强制刷新并声明我们要全权 (cloud-platform)
    # 这会覆盖掉任何本地缓存的低权限通行证
    if credentials.requires_scopes:
        credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])
    
    # 3. 强制刷新 Token
    auth_request = Request()
    credentials.refresh(auth_request)

    print(f"正在以身份 {credentials.service_account_email} 访问...")
    
    service = build('drive', 'v3', credentials=credentials)
    folder_id = '1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkm1CIK'

    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="files(id, name)").execute()
        
        items = results.get('files', [])
        print(f"🎉 成功！找到 {len(items)} 个文件：")
        for item in items:
            print(f"- {item['name']}")
    except Exception as e:
        print(f"❌ 依然报错，请检查 Google Drive API 是否已开启。")
        print(f"报错详情: {e}")

if __name__ == "__main__":
    run_test()
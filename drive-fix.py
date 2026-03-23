import google.auth
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import time

def run():
    # 1. 获取凭证
    credentials, project = google.auth.default()
    
    # 2. 强制刷新，解决你看到的“Email为空”问题
    print("正在唤醒虚拟机身份...")
    auth_request = Request()
    credentials.refresh(auth_request)
    
    # 3. 打印身份确认
    print(f"当前使用的邮箱: {credentials.service_account_email}")
    
    # 4. 强制指定 Scope (防止它拿旧的)
    if credentials.requires_scopes:
        credentials = credentials.with_scopes(['https://www.googleapis.com/auth/cloud-platform'])

    service = build('drive', 'v3', credentials=credentials)
    folder_id = '1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkm1CIK'

    try:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="files(id, name)").execute()
        items = results.get('files', [])
        print(f"🎉 彻底通了！文件夹里有 {len(items)} 个文件。")
    except Exception as e:
        print(f"❌ 还是不行。错误码: {e}")

if __name__ == "__main__":
    run()
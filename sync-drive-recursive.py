import os
from google.oauth2 import service_account
from googleapiclient.discovery import build

KEY_FILE = 'key.json'

creds = service_account.Credentials.from_service_account_file(
    KEY_FILE, scopes=['https://www.googleapis.com/auth/drive.readonly']
)
service = build('drive', 'v3', credentials=creds)

def debug_check():
    # 1. 确认机器人是谁
    print(f"👤 机器人账号: {creds.service_account_email}")
    
    # 2. 检查权限：直接尝试获取目标文件夹的元数据
    target_id = '1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkmlCIK'
    try:
        folder = service.files().get(fileId=target_id, fields="name, mimeType", supportsAllDrives=True).execute()
        print(f"📍 目标文件夹确认: 名称='{folder['name']}', 类型='{folder['mimeType']}'")
    except Exception as e:
        print(f"❌ 权限拒绝！机器人根本看不见 ID 为 {target_id} 的东西。报错: {e}")
        print("请检查：你是否真的把文件夹 Share 给了上面的机器人账号？")
        return

    # 3. 盲搜：列出机器人名下“所有”可见的项目
    print("\n🔍 正在全局搜索机器人能看到的所有内容...")
    results = service.files().list(
        pageSize=20, 
        fields="files(id, name, mimeType)",
        supportsAllDrives=True,
        includeItemsFromTrashed=False
    ).execute()
    
    items = results.get('files', [])
    if not items:
        print("🕳️ 搜索结果：空。机器人是“睁眼瞎”，什么都看不见。")
    else:
        print(f"✨ 成功！机器人一共能看见 {len(items)} 个项目。前几个是：")
        for item in items:
            print(f" - {item['name']} ({item['id']})")

if __name__ == "__main__":
    debug_check()
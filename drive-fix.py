import os
from google.oauth2 import service_account
from googleapiclient.discovery import build

# 1. 钥匙文件路径（确保 key.json 和这个脚本在同一个目录下）
KEY_FILE = 'key.json'
# 2. 你的文件夹 ID
FOLDER_ID = '1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkmlCIK'

def test_connection():
    # 检查文件是否存在
    if not os.path.exists(KEY_FILE):
        print(f"❌ 错误：在当前目录下没找到 {KEY_FILE} 文件！")
        return

    try:
        # 使用 JSON Key 授权
        print("正在使用 key.json 验证身份...")
        creds = service_account.Credentials.from_service_account_file(
            KEY_FILE, 
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        
        # 构造 Google Drive 服务
        service = build('drive', 'v3', credentials=creds)

        # 尝试列出文件夹内的文件
        print(f"正在读取文件夹 {FOLDER_ID} ...")
        results = service.files().list(
            q=f"'{FOLDER_ID}' in parents and trashed = false",
            fields="files(id, name)",
            pageSize=10
        ).execute()

        items = results.get('files', [])

        if not items:
            print("⚠️ 连接成功，但文件夹里没看到文件。请检查文件夹 ID 是否正确，并确认已分享给 Service Account 邮箱。")
        else:
            print(f"✅ 完美！成功列出前 {len(items)} 个文件：")
            for item in items:
                print(f" - {item['name']} (ID: {item['id']})")
            print("\n恭喜你，链路已经彻底打通了！")

    except Exception as e:
        print(f"❌ 还是出错了。详情如下：\n{e}")

if __name__ == "__main__":
    test_connection()
import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ================= 配置区 =================
KEY_FILE = 'key.json'  
# 再次确认 ID：倒数第 4 位是小写 L (lCIK)
ROOT_FOLDER_ID = '1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkmlCIK'  
LOCAL_ROOT = os.path.expanduser('~/coach-raw-video') 
# ==========================================

# 1. 授权
creds = service_account.Credentials.from_service_account_file(
    KEY_FILE, scopes=['https://www.googleapis.com/auth/drive.readonly']
)
service = build('drive', 'v3', credentials=creds)

def sync_folder(drive_folder_id, local_dir):
    # 确保本地文件夹存在
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        print(f"📁 已创建本地目录: {local_dir}")

    # 列出 Drive 文件夹下的所有内容
    results = service.files().list(
        q=f"'{drive_folder_id}' in parents and trashed = false",
        fields="files(id, name, mimeType)",
        pageSize=1000
    ).execute()
    
    items = results.get('files', [])
    print(f"🔎 正在扫描 Drive 目录，发现 {len(items)} 个项目...")

    for item in items:
        file_id = item['id']
        file_name = item['name']
        mime_type = item['mimeType']

        # A. 如果是子文件夹 -> 递归进入
        if mime_type == 'application/vnd.google-apps.folder':
            print(f"📂 发现子文件夹 [{file_name}]，正在递归进入...")
            sync_folder(file_id, os.path.join(local_dir, file_name))

        # B. 如果是文件 -> 统统下载
        else:
            dest_path = os.path.join(local_dir, file_name)
            
            # 如果本地已经有了，就跳过（支持断点续传）
            if os.path.exists(dest_path):
                print(f"⏩ 跳过已存在文件: {file_name}")
                continue

            print(f"📥 正在下载: {file_name} ...", end='', flush=True)
            try:
                request = service.files().get_media(fileId=file_id)
                fh = io.FileIO(dest_path, 'wb')
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                print(" [完成]")
            except Exception as e:
                # 针对 Google Docs/Sheets 等在线文档的特殊处理（它们不能直接下载，只能导出）
                if "only support direct download for binary content" in str(e):
                    print(" [跳过] (Google 在线文档无法直接下载，需手动导出)")
                else:
                    print(f" [错误] {e}")

if __name__ == "__main__":
    print(f"🚀 开始全量同步任务...")
    print(f"Drive 根 ID: {ROOT_FOLDER_ID}")
    print(f"本地目标: {LOCAL_ROOT}")
    
    try:
        sync_folder(ROOT_FOLDER_ID, LOCAL_ROOT)
        print("\n✨ 恭喜！同步任务全部完成。")
    except Exception as e:
        print(f"\n❌ 任务中断: {e}")
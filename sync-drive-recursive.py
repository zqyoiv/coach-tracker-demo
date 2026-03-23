import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ================= 配置区 =================
KEY_FILE = 'key.json'  # 你的密钥文件
ROOT_FOLDER_ID = '1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkm1CIK'  # 根文件夹 ID
LOCAL_ROOT = '~/coach-raw-video'  # 本地保存路径
# ==========================================

# 授权并构造服务
creds = service_account.Credentials.from_service_account_file(
    KEY_FILE, scopes=['https://www.googleapis.com/auth/drive.readonly']
)
service = build('drive', 'v3', credentials=creds)

def download_folder_recursive(folder_id, local_path):
    """
    递归进入 Drive 文件夹并下载视频
    """
    # 1. 确保本地目录存在
    if not os.path.exists(local_path):
        os.makedirs(local_path)
        print(f"📁 创建文件夹: {local_path}")

    # 2. 列出当前 Drive 文件夹下的所有内容 (包括文件和子文件夹)
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        fields="files(id, name, mimeType)",
        pageSize=1000
    ).execute()
    
    items = results.get('files', [])

    for item in items:
        file_id = item['id']
        file_name = item['name']
        mime_type = item['mime_type'] # 注意：API 返回的是 mimeType，这里对应 item['mimeType']

        # A. 如果是文件夹 -> 递归进入
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            new_local_path = os.path.join(local_path, file_name)
            download_folder_recursive(file_id, new_local_path)

        # B. 如果是视频文件 -> 下载
        elif 'video' in item['mimeType']:
            dest_file_path = os.path.join(local_path, file_name)
            
            # 跳过已存在的文件
            if os.path.exists(dest_file_path):
                print(f"⏩ 跳过已存在: {file_name}")
                continue

            print(f"📥 正在下载: {file_name} -> {local_path} ...", end='', flush=True)
            
            try:
                request = service.files().get_media(fileId=file_id)
                fh = io.FileIO(dest_file_path, 'wb')
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                print(" [完成]")
            except Exception as e:
                print(f" [失败] 错误: {e}")

def main():
    print(f"🚀 开始递归同步任务...")
    print(f"根目录 ID: {ROOT_FOLDER_ID}")
    download_folder_recursive(ROOT_FOLDER_ID, LOCAL_ROOT)
    print("\n✅ 所有任务已处理完毕！")

if __name__ == "__main__":
    main()
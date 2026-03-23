import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ================= 配置区 =================
KEY_FILE = 'key.json'  
# 注意：最后四位是 lCIK (小写字母 L)，不是 1CIK
ROOT_FOLDER_ID = '1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkmlCIK'  
# 建议使用绝对路径，或者 expanduser 来处理 ~ 符号
LOCAL_ROOT = os.path.expanduser('~/coach-raw-video') 
# ==========================================

creds = service_account.Credentials.from_service_account_file(
    KEY_FILE, scopes=['https://www.googleapis.com/auth/drive.readonly']
)
service = build('drive', 'v3', credentials=creds)

def download_folder_recursive(folder_id, local_path):
    if not os.path.exists(local_path):
        os.makedirs(local_path)
        print(f"📁 创建文件夹: {local_path}")

    # 列出内容
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        fields="files(id, name, mimeType)", # 注意这里是 mimeType
        pageSize=1000
    ).execute()
    
    items = results.get('files', [])

    for item in items:
        file_id = item['id']
        file_name = item['name']
        mime_type = item.get('mimeType', '')

        # A. 如果是文件夹
        if mime_type == 'application/vnd.google-apps.folder':
            new_local_path = os.path.join(local_path, file_name)
            download_folder_recursive(file_id, new_local_path)

        # B. 如果是视频文件
        elif 'video' in mime_type:
            dest_file_path = os.path.join(local_path, file_name)
            
            if os.path.exists(dest_file_path):
                print(f"⏩ 跳过: {file_name}")
                continue

            print(f"📥 下载中: {file_name}...", end='', flush=True)
            
            try:
                request = service.files().get_media(fileId=file_id)
                fh = io.FileIO(dest_file_path, 'wb')
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                print(" [完成]")
            except Exception as e:
                print(f" [失败] {e}")

def main():
    print(f"🚀 开始递归同步...")
    print(f"目标 ID: {ROOT_FOLDER_ID}")
    try:
        download_folder_recursive(ROOT_FOLDER_ID, LOCAL_ROOT)
        print("\n✅ 所有视频已同步至 ~/coach-raw-video")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        print("提示：请检查 key.json 对应的 Service Account 是否已被添加为文件夹的 'Viewer' 或 'Editor'")

if __name__ == "__main__":
    main()
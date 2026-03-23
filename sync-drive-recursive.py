import os
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ================= 配置区 =================
KEY_FILE = 'key.json'  
ROOT_FOLDER_ID = '1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkmlCIK'  
LOCAL_ROOT = os.path.expanduser('~/coach-raw-video') 
# ==========================================

creds = service_account.Credentials.from_service_account_file(
    KEY_FILE, scopes=['https://www.googleapis.com/auth/drive.readonly']
)
service = build('drive', 'v3', credentials=creds)

def download_recursive(folder_id, local_path):
    if not os.path.exists(local_path):
        os.makedirs(local_path, exist_user=True)
    
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        fields="files(id, name, mimeType)",
        pageSize=1000
    ).execute()
    
    items = results.get('files', [])
    print(f"\n📂 进入文件夹: {local_path} (发现 {len(items)} 个项目)")

    for item in items:
        fid, fname, ftype = item['id'], item['name'], item['mimeType']
        
        # 1. 如果是文件夹 -> 递归
        if ftype == 'application/vnd.google-apps.folder':
            print(f"  > 发现子文件夹: {fname}，正在进入...")
            download_recursive(fid, os.path.join(local_path, fname))
        
        # 2. 如果是文件 -> 强制下载 (只要不是文件夹统统下载)
        else:
            dest_path = os.path.join(local_path, fname)
            
            # 这里的判断改松了：只要名字以常见视频后缀结尾，或者类型里带 video
            is_video = 'video' in ftype or fname.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))
            
            if is_video:
                if os.path.exists(dest_path):
                    print(f"  - [跳过] {fname} (已存在)")
                    continue
                
                print(f"  - [下载] {fname} ({ftype})...", end='', flush=True)
                try:
                    request = service.files().get_media(fileId=fid)
                    fh = io.FileIO(dest_path, 'wb')
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                    print(" [OK]")
                except Exception as e:
                    print(f" [失败: {e}]")
            else:
                print(f"  - [忽略] {fname} (类型为 {ftype}，不匹配视频格式)")

if __name__ == "__main__":
    print("🚀 启动强制同步...")
    download_recursive(ROOT_FOLDER_ID, LOCAL_ROOT)
    print("\n✅ 任务结束。")
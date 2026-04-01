import os
import io
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ================= 配置区 =================
KEY_FILE = 'key.json'
ROOT_FOLDER_ID = '1zuHPXlu3oLNYC5Ri2LQ5qA7-_vkmlCIK'
LOCAL_ROOT = os.path.expanduser('~/coach-raw-video')
# Only this subfolder of ROOT (and its descendants) is synced — not sibling date folders.
SYNC_FOLDER_NAME = '3-23'
# ==========================================

creds = service_account.Credentials.from_service_account_file(
    KEY_FILE, scopes=['https://www.googleapis.com/auth/drive.readonly']
)
service = build('drive', 'v3', credentials=creds)


def find_child_folder_id(parent_id: str, name: str) -> Optional[str]:
    """Return Drive folder id for direct child of parent_id with exact name, or None."""
    # Escape single quotes for Drive query (name is fixed literal here)
    safe = name.replace("'", "\\'")
    q = (
        f"'{parent_id}' in parents and trashed = false "
        f"and mimeType = 'application/vnd.google-apps.folder' "
        f"and name = '{safe}'"
    )
    token = None
    while True:
        req = service.files().list(
            q=q,
            fields="nextPageToken, files(id, name)",
            pageSize=100,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            pageToken=token,
        )
        res = req.execute()
        for f in res.get('files', []):
            if f.get('name') == name:
                return f['id']
        token = res.get('nextPageToken')
        if not token:
            break
    return None


def download_recursive(folder_id, local_path):
    # 1. 创建本地目录
    if not os.path.exists(local_path):
        os.makedirs(local_path, exist_ok=True)
        print(f"📁 创建文件夹: {local_path}")

    # 2. 获取该文件夹下的内容
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed = false",
        fields="files(id, name, mimeType)",
        pageSize=1000,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True
    ).execute()
    
    items = results.get('files', [])
    
    for item in items:
        fid, fname, ftype = item['id'], item['name'], item['mimeType']
        
        # A. 如果是子文件夹 -> 继续钻
        if ftype == 'application/vnd.google-apps.folder':
            download_recursive(fid, os.path.join(local_path, fname))
            
        # B. 如果是文件 -> 直接下载
        else:
            dest_path = os.path.join(local_path, fname)
            if os.path.exists(dest_path):
                print(f"⏩ 跳过已存在: {fname}")
                continue

            print(f"📥 正在下载: {fname}...", end='', flush=True)
            try:
                request = service.files().get_media(fileId=fid)
                fh = io.FileIO(dest_path, 'wb')
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                print(" [OK]")
            except Exception as e:
                print(f" [跳过] {e}")

if __name__ == "__main__":
    local_target = os.path.join(LOCAL_ROOT, SYNC_FOLDER_NAME)
    print(f"🚀 只同步 Drive 子文件夹「{SYNC_FOLDER_NAME}」到: {local_target}")
    try:
        sub_id = find_child_folder_id(ROOT_FOLDER_ID, SYNC_FOLDER_NAME)
        if not sub_id:
            print(
                f"❌ 在根目录下未找到名为「{SYNC_FOLDER_NAME}」的文件夹（ROOT_FOLDER_ID 的直接子项）。"
            )
            raise SystemExit(1)
        download_recursive(sub_id, local_target)
        print("\n✅ 同步圆满完成！")
    except SystemExit:
        raise
    except Exception as e:
        print(f"\n❌ 运行中断: {e}")
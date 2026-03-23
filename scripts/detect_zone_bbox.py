import cv2
import numpy as np
import sys


IMG_PATH = r"C:\Users\vioyq\.cursor\projects\c-Users-vioyq-Desktop-Coach-Tracker-coach-tracker-demo\assets\c__Users_vioyq_AppData_Roaming_Cursor_User_workspaceStorage_d281b35197440e827032053337577d9c_images_image-4d4bbce5-7b62-4b46-a7da-65e0c820c6fa.png"


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else IMG_PATH
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Could not read image: {path}")

    h, w = img.shape[:2]
    b, g, r = cv2.split(img)
    # Stricter magenta threshold for the drawn rectangle border.
    mask = ((r > 180) & (b > 180) & (g < 160)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise SystemExit("No magenta pixels detected.")

    # Ignore right-side UI controls.
    valid = (xs > int(0.05 * w)) & (xs < int(0.85 * w)) & (ys > int(0.05 * h)) & (ys < int(0.99 * h))
    xs = xs[valid]
    ys = ys[valid]
    if len(xs) == 0:
        raise SystemExit("No valid magenta pixels after filtering.")

    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())

    # Debug strongest horizontal/vertical magenta lines.
    row_sum = (mask > 0).sum(axis=1)
    col_sum = (mask > 0).sum(axis=0)
    top_rows = np.argsort(row_sum)[-8:][::-1]
    top_cols = np.argsort(col_sum)[-8:][::-1]

    print("image_size", w, h)
    print("bbox_px", x1, y1, x2, y2)
    print("ZONE_NORM", round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4))
    print("top_rows", [(int(i), int(row_sum[i])) for i in top_rows])
    print("top_cols", [(int(i), int(col_sum[i])) for i in top_cols])


if __name__ == "__main__":
    main()


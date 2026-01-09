# step12_make_scatter_mosaic.py
# ---------------------------------------------------------
# 功能：将 data/DVcompare/<case_name>/ 下的 7 张散点图（raw + 6个η）
#      按指定行列拼成一张大图，默认拼成 1×7，用于论文排版。
#
# 使用方式：
# - 直接在本文件顶部修改 CASE_DIR（指向某个对比结果文件夹）
# - 可修改 ROWS/COLS 控制拼图行列，默认 1×7
# ---------------------------------------------------------

import os
from PIL import Image


# -------------------------------
# 你只需要改这里
# -------------------------------
CASE_DIR = 'data/DVcompare/On_set2_trail2'
# CASE_DIR = 'data/DVcompare/Off_set3_trail5'
# ROWS = 1
# COLS = 7
ROWS = 2
COLS = 4

# 文件名规则（与你前面脚本一致）
RAW_NAME = "raw_scatter.png"
ETA_NAMES = [
    "eta_0.050_scatter.png",
    "eta_0.100_scatter.png",
    "eta_0.150_scatter.png",
    "eta_0.200_scatter.png",
    "eta_0.250_scatter.png",
    "eta_0.300_scatter.png",
    'eta_0.150_removed_scatter.png'
]

OUT_NAME = "mosaic_2x4.png"  # 默认输出名；如果你改ROWS/COLS，建议同步改名


def _load_images(paths):
    imgs = []
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing image: {p}")
        imgs.append(Image.open(p).convert("RGB"))
    return imgs


def _resize_to_same_height(imgs):
    """
    把所有图片缩放到相同高度（取最小高度），宽度按比例缩放，
    避免拼图时出现大小不一致导致对齐问题。
    """
    min_h = min(im.height for im in imgs)
    out = []
    for im in imgs:
        if im.height == min_h:
            out.append(im)
        else:
            new_w = int(round(im.width * (min_h / im.height)))
            out.append(im.resize((new_w, min_h), Image.LANCZOS))
    return out


def make_mosaic(imgs, rows, cols, bg_color=(255, 255, 255), pad=10):
    """
    把imgs按 rows×cols 拼成一张大图。
    - imgs数量不足：后面空位留白
    - imgs数量超过：只取前 rows*cols 张
    """
    n = rows * cols
    imgs = imgs[:n]

    imgs = _resize_to_same_height(imgs)

    # 每一列的最大宽度（因为按比例缩放后宽度可能不同）
    col_widths = [0] * cols
    for i, im in enumerate(imgs):
        c = i % cols
        col_widths[c] = max(col_widths[c], im.width)

    # 每一行的高度（现在都一样高）
    cell_h = imgs[0].height if imgs else 0
    row_heights = [cell_h] * rows

    total_w = sum(col_widths) + pad * (cols + 1)
    total_h = sum(row_heights) + pad * (rows + 1)

    canvas = Image.new("RGB", (total_w, total_h), bg_color)

    # 粘贴
    idx = 0
    y = pad
    for r in range(rows):
        x = pad
        for c in range(cols):
            if idx < len(imgs):
                im = imgs[idx]
                # 居中贴到该cell
                x_off = x + (col_widths[c] - im.width) // 2
                canvas.paste(im, (x_off, y))
                idx += 1
            x += col_widths[c] + pad
        y += row_heights[r] + pad

    return canvas


def main():
    if not os.path.isdir(CASE_DIR):
        raise FileNotFoundError(f"Missing CASE_DIR: {CASE_DIR}")

    img_paths = [os.path.join(CASE_DIR, RAW_NAME)] + [os.path.join(CASE_DIR, n) for n in ETA_NAMES]
    imgs = _load_images(img_paths)

    mosaic = make_mosaic(imgs, ROWS, COLS, pad=10)

    out_path = os.path.join(CASE_DIR, OUT_NAME)
    mosaic.save(out_path)
    print(f"[DONE] saved mosaic: {out_path}")


if __name__ == "__main__":
    main()

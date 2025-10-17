import cv2
import numpy as np
import pandas as pd
import easyocr
import re
import argparse


def load_image(file_path):
    img_read = cv2.imread(file_path)
    if img_read is None:
        raise Exception("Please check the image file.")
    return img_read

def preprocess_table(img):
    # 1. Rectangle detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    table_quad = None
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(cnt) > max_area:
            max_area = cv2.contourArea(cnt)
            table_quad = approx

    if table_quad is None:
        raise Exception("No Rectangle found.")


    # 2. Vertex alignment
    pts = table_quad.reshape(4, 2)
    sum_pts = pts.sum(axis=1)
    diff_pts = np.diff(pts, axis=1).reshape(-1)
    ordered_pts = np.array([
        pts[np.argmin(sum_pts)],      # Top-left
        pts[np.argmin(diff_pts)],     # Top-right
        pts[np.argmax(sum_pts)],      # Bottom-right
        pts[np.argmax(diff_pts)]      # Bottom-left
    ], dtype='float32')


    # 3. Crop only the table with perspective transformation
    w1 = np.linalg.norm(ordered_pts[0] - ordered_pts[1])
    w2 = np.linalg.norm(ordered_pts[2] - ordered_pts[3])
    h1 = np.linalg.norm(ordered_pts[0] - ordered_pts[3])
    h2 = np.linalg.norm(ordered_pts[1] - ordered_pts[2])
    width = int(max(w1, w2))
    height = int(max(h1, h2))

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype='float32')

    perspective = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
    warped = cv2.warpPerspective(img, perspective, (width, height))


    # 4. Rotate in a long horizontal shape
    if height > width:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)


    # 5. Rotate to position the dark part at the top.
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    h2, w2 = warped_gray.shape
    top_mean = np.mean(warped_gray[:int(h2 * 0.15), :])
    bottom_mean = np.mean(warped_gray[int(h2 * 0.85):, :])

    if top_mean <= bottom_mean:
        final = warped_gray
    else:
        final = cv2.rotate(warped_gray, cv2.ROTATE_180)

    return final


def post_process_currency(text_data_list):
    currency_patterns = r"([YI16S])"
    number_pattern = r"(\d{1,3}(?:,\d{3})*(?:\.\d+)?)"
    regex = re.compile(r"(?<![\u3040-\u30ff\u4e00-\u9faf])" + currency_patterns + r"\s*" + number_pattern)

    processed_data_list = []
    for item in text_data_list:
        text = item['text']
        new_text = regex.sub(r"¥\2", text)
        item['text'] = new_text
        processed_data_list.append(item)

    return processed_data_list


def recognize_and_grid(rotated_gray_img):
    # 1. Optical Character Recognition
    reader = easyocr.Reader(['ja', 'en'])
    results = reader.readtext(rotated_gray_img, detail=1)

    texts_data = []
    for (bbox, text, conf) in results:
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[2]
        h = y_max - y_min

        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        texts_data.append({
            'x': center_x,
            'y': center_y,
            'h': h,
            'text': text,
            'bbox': (bbox[0], bbox[2])
        })

    if not texts_data:
        raise Exception("No text recognized.")

    texts_data = post_process_currency(texts_data)

    # 2. Row clustering
    texts_data.sort(key=lambda item: item['y'])

    all_heights = [item['h'] for item in texts_data]
    avg_text_height = np.median(all_heights)
    row_separator_threshold = avg_text_height * 1.5

    y_coords = np.array([item['y'] for item in texts_data])
    y_diffs = np.diff(y_coords)

    new_row_indices = np.where(y_diffs > row_separator_threshold)[0] + 1
    rows = np.split(texts_data, new_row_indices)
    rows = [list(row) for row in rows if len(row) > 0]

    if not rows:
        raise Exception("Row split failed.")

    # 3. Column clustering and grid configuration
    for row in rows:
        row.sort(key=lambda item: item['x'])

    pivot_row = max(rows, key=len)
    column_x_centers = [item['x'] for item in pivot_row]
    max_cols = len(column_x_centers)
    data = []

    for row in rows:
        row_data = [''] * max_cols
        for item in row:
            x_pos = item['x']
            distances = [abs(x_pos - center) for center in column_x_centers]
            closest_col_index = np.argmin(distances)

            if row_data[closest_col_index] == '':
                row_data[closest_col_index] = item['text']
            else:
                row_data[closest_col_index] += ' ' + item['text']

        data.append(row_data)

    df = pd.DataFrame(data)

    return df


def main(input_image, output_csv):
    try:
        load_img = load_image(input_image)
        table_img = preprocess_table(load_img)
        result_df = recognize_and_grid(table_img)
        result_df.to_csv(output_csv, index=False, header=False, encoding='utf-8-sig')
        print(result_df.to_string(index=False, header=False))
    except Exception as e:
        print(f"[ERROR] {e}")
        exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OCR Table Extractor')
    parser.add_argument('-i', '--input', required=False, help='Input image file path')
    parser.add_argument('-o', '--output', required=False, help='Output CSV file path')
    args = parser.parse_args()

    if not args.input or not args.output:
        print("[ERROR] Both -i (input image) and -o (output csv) options are required.")
        print("Example: python script.py -i xxx.png -o yyy.csv")
        exit(1)

    main(args.input, args.output)

# Mask-2-JSON-TXT
Mask labeled tataset chang style to JSON &amp; TXT
# Background

将mask数据转换为json数据以便检查是否正确，再将json数据转换为YOLOv8训练可以使用的txt文件。

# Mask2JSON

1. 转换需要从掩膜图片中读取信息，依次写入json文件中，需要用到opencv, shapely, 尤其是shapely, 

```python
pip install opencv-python-headless shapely
```

1. 由于写入json的imagePath会默认为json路径，这是不正确的，会导致生成的json文件由于索引错误无法打开，所以需要专门指定：

```python
    json_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path), # 这里需要专门指定。
        "imageData": image_data,
        "imageHeight": image_height,
        "imageWidth": image_width
    }
```

1. 由于需要通过查看json文件判断图片的标注是否正确，所以在此阶段不需要把坐标归一化。
2. 由于掩膜操作的不确定性，可能存在多边形不闭合，两个点在同一个地方，点交叉从而导致在判断polygons的时候将某一些标注判断为false,所以需要以下处理：

```python
def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for i, contour in enumerate(contours):
        if len(contour) >= 3:
            polygon = Polygon(contour.squeeze(1))
            if not polygon.is_valid:
                print(f'Contour {i} is invalid. Attempting to fix it.')
                polygon = polygon.buffer(0)
            if isinstance(polygon, MultiPolygon):
                print(f'Contour {i} resulted in a MultiPolygon. Processing each sub-polygon separately.')
                for sub_polygon in polygon.geoms:  # Use polygon.geoms to iterate over sub-polygons
                    if sub_polygon.is_valid:
                        polygons.append(sub_polygon)
                        print(f'Sub-polygon is valid, Points={len(sub_polygon.exterior.coords)}')
                    else:
                        print(f'Sub-polygon is still invalid, Points={len(sub_polygon.exterior.coords)}')
            elif polygon.is_valid:
                polygons.append(polygon)
                print(f'Contour {i}: Valid=True, Points={len(contour)}')
            else:
                print(f'Contour {i}: Still invalid after buffer, Points={len(contour)}')
    return polygons
```

1. 有一些encode过后的图片会隐藏图片信息，作为null写入json文件后会导致缺乏关键信息使得json文件无法打开，所以需要解码操作，虽然会导致json文件变大，但是如果遇到a bytes-like object is required, not 'NoneType’的时候，可以尝试该操作：

```python
def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')
    return image_data
```

1. 有一些polygons由于无法组成多边形会造成之后转txt文件时segmentation后面的坐标信息为空，所以需要修改以下判断操作：

```python
def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for i, contour in enumerate(contours):
        if len(contour) >= 3:
            polygon = Polygon(contour.squeeze(1))
            if not polygon.is_valid:
                print(f'Contour {i} is invalid. Attempting to fix it.')
                polygon = polygon.buffer(0)
            if isinstance(polygon, MultiPolygon):
                print(f'Contour {i} resulted in a MultiPolygon. Processing each sub-polygon separately.')
                for sub_polygon in polygon.geoms:
                    if sub_polygon.is_valid:
                        polygons.append(sub_polygon)
                        print(f'Sub-polygon is valid, Points={len(sub_polygon.exterior.coords)}')
                    else:
                        print(f'Sub-polygon is still invalid, Points={len(sub_polygon.exterior.coords)}')
            elif polygon.is_valid:
                polygons.append(polygon)
                print(f'Contour {i}: Valid=True, Points={len(contour)}')
            else:
                print(f'Contour {i}: Still invalid after buffer, Points={len(contour)}')
    return polygons
    
    
    def save_to_json(polygons, json_path, image_path, image_width, image_height):
    shapes = []
    for i, polygon in enumerate(polygons):
        if polygon.is_valid:
            points = [(float(x), float(y)) for x, y in polygon.exterior.coords]
            if points:  # 检查 points 列表是否为空
                shape = {
                    "label": "segmentation",
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                print(f'Polygon {i}: Points={len(points)}')
                shapes.append(shape)
                
                ......
```

## Full coding

完整处理代码如下：

```python
import os
import cv2
import numpy as np
import json
from shapely.geometry import Polygon, MultiPolygon
import base64

def mask_to_polygons(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for i, contour in enumerate(contours):
        if len(contour) >= 3:
            polygon = Polygon(contour.squeeze(1))
            if not polygon.is_valid:
                print(f'Contour {i} is invalid. Attempting to fix it.')
                polygon = polygon.buffer(0)
            if isinstance(polygon, MultiPolygon):
                print(f'Contour {i} resulted in a MultiPolygon. Processing each sub-polygon separately.')
                for sub_polygon in polygon.geoms:
                    if sub_polygon.is_valid:
                        polygons.append(sub_polygon)
                        print(f'Sub-polygon is valid, Points={len(sub_polygon.exterior.coords)}')
                    else:
                        print(f'Sub-polygon is still invalid, Points={len(sub_polygon.exterior.coords)}')
            elif polygon.is_valid:
                polygons.append(polygon)
                print(f'Contour {i}: Valid=True, Points={len(contour)}')
            else:
                print(f'Contour {i}: Still invalid after buffer, Points={len(contour)}')
    return polygons

def encode_image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')
    return image_data

def save_to_json(polygons, json_path, image_path, image_width, image_height):
    shapes = []
    for i, polygon in enumerate(polygons):
        if polygon.is_valid:
            points = [(float(x), float(y)) for x, y in polygon.exterior.coords]
            if points:  # 检查 points 列表是否为空
                shape = {
                    "label": "segmentation",
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                print(f'Polygon {i}: Points={len(points)}')
                shapes.append(shape)

    image_data = encode_image_to_base64(image_path)

    json_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": image_data,
        "imageHeight": image_height,
        "imageWidth": image_width
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)

def process_directory(mask_dir, json_dir):
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png') or f.endswith('.bmp')]

    for cnt, mask_name in enumerate(mask_files):
        print(f'Processing file {cnt + 1}/{len(mask_files)}: {mask_name}')
        mask_path = os.path.join(mask_dir, mask_name)
        json_path = os.path.join(json_dir, mask_name.replace('.png', '.json').replace('.bmp', '.json'))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Error reading {mask_path}")
            continue

        image_height, image_width = mask.shape
        polygons = mask_to_polygons(mask)
        print(f'Found {len(polygons)} polygons')  # Debugging line
        save_to_json(polygons, json_path, mask_path, image_width, image_height)

if __name__ == "__main__":
    mask_dir = r"D:\work\DATA\analysis_data\TmaTrace_07_all_InstanceSeg\Annotations"
    json_dir = r"D:\work\DATA\analysis_data\TmaTrace_07_all_InstanceSeg\TmaTrace_07_all_INstanceSeg_json"

    process_directory(mask_dir, json_dir)
```

# JSON2TXT

具体步骤详见：[Labelme_JSON 2 TXT](https://www.notion.so/Labelme_JSON-2-TXT-e5cb1b99fadc4f168cf0880a71169b4a?pvs=21)

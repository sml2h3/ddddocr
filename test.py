import ddddocr

# 识别
ocr = ddddocr.DdddOcr(old=True)

with open('test.jpg', 'rb') as f:
    img_bytes = f.read()

res = ocr.classification(img_bytes)
print(res)

# 目标检测
det = ddddocr.DdddOcr(det=True)

with open('test.jpg', 'rb') as f:
    img_bytes = f.read()

res = det.detection(img_bytes)
print(res)
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

# 滑块模板匹配方式

det = ddddocr.DdddOcr(det=False, ocr=False)

with open('b.png', 'rb') as f:
    target_bytes = f.read()

with open('a.png', 'rb') as f:
    background_bytes = f.read()

res = det.slide_match(target_bytes, background_bytes)
print(res)

det = ddddocr.DdddOcr(det=False, ocr=False)

with open('bg.jpg', 'rb') as f:
    target_bytes = f.read()

with open('fullpage.jpg', 'rb') as f:
    background_bytes = f.read()


res = det.slide_comparison(target_bytes, background_bytes)

print(res)
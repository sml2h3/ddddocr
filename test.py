import ddddocr

ocr = ddddocr.DdddOcr()

with open('test.png', 'rb') as f:
    img_bytes = f.read()

res = ocr.classification(img_bytes)
print(res)
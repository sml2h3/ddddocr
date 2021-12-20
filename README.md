![header.png](https://z3.ax1x.com/2021/07/02/R6Ih28.jpg)

# 带带弟弟OCR通用验证码识别SDK免费开源版

## 交流群（找对象，在苏州，dd群主）


![qrcode.png](http://cdn.wenanzhe.com/mmqrcode1639968409395.png)

## 环境要求

`python >= 3.8`

`Windows/Linux..`

## 调用方法

`pip install ddddocr`

```
import ddddocr

ocr = ddddocr.DdddOcr()

with open('test.png', 'rb') as f:

    img_bytes = f.read()

res = ocr.classification(img_bytes)

print(res)
```

### 参数说明

`DdddOcr 接受两个参数`

|  参数名   | 默认值  | 说明  |
|  ----  | ----  | ----  |
| use_gpu  | False | Bool    是否使用gpu进行推理，如果该值为False则device_id不生效 |
| device_id  | 0 | int cuda设备号，目前仅支持单张显卡 |

`classification`

|  参数名   | 默认值  | 说明  |
|  ----  | ----  | ----  |
| img  | 0 | bytes 图片的bytes格式 |

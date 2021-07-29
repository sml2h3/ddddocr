![header.png](https://z3.ax1x.com/2021/07/02/R6Ih28.jpg)

# 带带弟弟OCR通用验证码识别SDK免费开源版

## 环境要求

`python >= 3.8`

`Windows/Linux/Macox..`

## 调用方法

`pip install ddddocr`

```python
import ddddocr
ocr = ddddocr.DdddOcr()
with open('test.png', 'rb') as f:
    img_bytes = f.read()
res = ocr.classification(img_bytes=img_bytes)
print(res)
```
或者传入图片 base64 编码值（不包含图片头）
```python
import ddddocr
ocr = ddddocr.DdddOcr()
img_base64 = 'img_base64' # 示例
res = ocr.classification(img_base64=img_base64)
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
| img_bytes  | None | bytes 图片的bytes格式 |
| img_base64  | None | 图片的 base64 编码值（不包含图片头） |

> 说明，当 `img_bytes` 和 `img_base64` 都存在时，优先使用 `img_bytes`
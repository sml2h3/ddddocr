![header.png](https://z3.ax1x.com/2021/07/02/R6Ih28.jpg)

# 带带弟弟OCR通用验证码识别SDK免费开源版

# 2021/12/24重大更新，ddddocr现在支持通用目标检测啦


## 交流群（加我微信拉你进群）

![qrcode.png](https://cdn.wenanzhe.com/img/mmqrcode1640418911274(1).png)

## 环境要求

`python <= 3.9`

`Windows/Linux/Macos..`

暂时不支持Macbook M1(X)，M1(X)用户需要自己编译onnxruntime才可以使用

## 调用方法

`pip install ddddocr`

### 1、文字识别模式

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

### 2、目标检测模式
```python
import ddddocr
det = ddddocr.DdddOcr(det=True)

with open('test.jpg', 'rb') as f:
    img_bytes = f.read()

res = det.detection(img_bytes)
print(res)
```

### 3、参数说明

`DdddOcr 接受三个参数`

|  参数名   | 默认值  | 说明  |
|  ----  | ----  | ----  |
| det  | False | Bool 默认为识别文字模式，为True则开启目标检测模式 |
| use_gpu  | False | Bool    是否使用gpu进行推理，如果该值为False则device_id不生效 |
| device_id  | 0 | int cuda设备号，目前仅支持单张显卡 |

`classification`

必须det参数为False后才可使用

|  参数名   | 默认值  | 说明  |
|  ----  | ----  | ----  |
| img_bytes  | None | bytes 图片的bytes格式 |
| img_base64  | None | 图片的 base64 编码值（不包含图片头） |

`detection`

必须det参数为False后才可使用

|  参数名   | 默认值  | 说明  |
|  ----  | ----  | ----  |
| img_bytes  | None | bytes 图片的bytes格式 |
| img_base64  | None | 图片的 base64 编码值（不包含图片头） |

> 说明，当 `img_bytes` 和 `img_base64` 都存在时，优先使用 `img_bytes`

> 如果使用GPU，需要自行安装cuda和cudnn，并在安装完ddddocr时执行 <br>`pip uninstall onnxrumtime`<br>然后手动执行<br>`pip install onnxruntime-gpu`

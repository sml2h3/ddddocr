![header.png](https://z3.ax1x.com/2021/07/02/R6Ih28.jpg)

# 带带弟弟OCR通用验证码识别SDK免费开源版




# 当前版本为1.4.8

## 1.4.3更新内容

本次升级的主要原因为，[dddd_trainer](https://github.com/sml2h3/dddd_trainer) 的开源进行适配，使[dddd_trainer](https://github.com/sml2h3/dddd_trainer) 训练出的模型可以直接无缝导入到ddddocr里面来使用

### 支持使用ddddocr调用 [dddd_trainer](https://github.com/sml2h3/dddd_trainer) 训练后的自定义模型

[dddd_trainer](https://github.com/sml2h3/dddd_trainer) 训练后会在models目录里导出charsets.json和onnx模型

如下所示，import_onnx_path为onnx所在地址，charsets_path为onnx所在地址
```python
import ddddocr

ocr = ddddocr.DdddOcr(det=False, ocr=False, import_onnx_path="myproject_0.984375_139_13000_2022-02-26-15-34-13.onnx", charsets_path="charsets.json")

with open('888e28774f815b01e871d474e5c84ff2.jpg', 'rb') as f:
    image_bytes = f.read()

res = ocr.classification(image_bytes)
print(res)

```

# 捐赠 （如果项目有帮助到您，可以选择捐赠一些费用用于ddddocr的后续版本维护，本项目长期维护）

 ![Test](https://cdn.wenanzhe.com/img/zhifubao.jpg!/scale/30) 
 ![Test](https://cdn.wenanzhe.com/img/weixin.jpg!/scale/30)

# 赞助合作商

|                                                            | 赞助合作商 | 推荐理由                                                                                             |
|------------------------------------------------------------|------------|--------------------------------------------------------------------------------------------------|
| ![YesCaptcha](https://cdn.wenanzhe.com/img/yescaptcha.png) | [YesCaptcha](https://yescaptcha.com/i/NSwk7i) | 谷歌reCaptcha验证码 / hCaptcha验证码 / funCaptcha验证码商业级识别接口 [点我](https://yescaptcha.com/i/NSwk7i) 直达VIP4 |

# 1.4.0版本更新内容

  本次更新新增了两种滑块识别算法，算法非深度神经网络实现，仅使用opencv和PIL完成。

  ## 算法1
  小滑块为单独的png图片，背景是透明图，如下图

  ![Test](https://cdn.wenanzhe.com/img/b.png) 

  然后背景为带小滑块坑位的，如下图 
  
  ![Test](https://cdn.wenanzhe.com/img/a.png) 

  ```python
    det = ddddocr.DdddOcr(det=False, ocr=False)
    
    with open('target.png', 'rb') as f:
        target_bytes = f.read()
    
    with open('background.png', 'rb') as f:
        background_bytes = f.read()
    
    res = det.slide_match(target_bytes, background_bytes)
    
    print(res)
  ```
  *提示：如果小图无过多背景部分，则可以添加simple_target参数， 通常为jpg或者bmp格式的图片*
```python
    slide = ddddocr.DdddOcr(det=False, ocr=False)
    
    with open('target.jpg', 'rb') as f:
        target_bytes = f.read()
    
    with open('background.jpg', 'rb') as f:
        background_bytes = f.read()
    
    res = slide.slide_match(target_bytes, background_bytes, simple_target=True)
    
    print(res)
  ```
  ## 算法2
  一张图为带坑位的原图，如下图

  ![Test](https://cdn.wenanzhe.com/img/bg.jpg) 

  一张图为原图，如下图 
  
  ![Test](https://cdn.wenanzhe.com/img/fullpage.jpg) 

  ```python
    slide = ddddocr.DdddOcr(det=False, ocr=False)

    with open('bg.jpg', 'rb') as f:
        target_bytes = f.read()
    
    with open('fullpage.jpg', 'rb') as f:
        background_bytes = f.read()
    
    img = cv2.imread("bg.jpg")
    
    res = slide.slide_comparison(target_bytes, background_bytes)

    print(res)
  ```

  ## 更新内容2
  添加全局ocr关闭参数，初始化时传入

 `dddd = ddddocr.DdddOcr(ocr=False)`

  则为关闭ocr功能，如果det = True，则会自动关闭ocr


# 1.3.1版本更新内容

  想必很多做验证码的新手，一定头疼碰到点选类型的图像，做样本费时费力，神经网络不会写，训练设备太昂贵，模型效果又不好。

  市场上常见的点选类验证码图片如下图所示


 ![Test](https://cdn.wenanzhe.com/img/0446fe794381489f90719d5e0506f2da.jpg) 

 ![Test](https://cdn.wenanzhe.com/img/6175e944c1dc408a89aabe4f7fc07fca.jpg) 

 ![Test](https://cdn.wenanzhe.com/img/20211226135747.png) 

  ![Test](https://cdn.wenanzhe.com/img/f34390d4911c45ce9058dc2e7e9d847a.jpg) 

  那么今天，他来了，ddddocr带着重磅更新大摇大摆的走来了。
# 简介
  ddddocr是由sml2h3开发的专为验证码厂商进行对自家新版本验证码难易强度进行验证的一个python库，其由作者与kerlomz共同合作完成，通过大批量生成随机数据后进行深度网络训练，本身并非针对任何一家验证码厂商而制作，本库使用效果完全靠玄学，可能可以识别，可能不能识别。

  ddddocr奉行着开箱即用、最简依赖的理念，尽量减少用户的配置和使用成本，希望给每一位测试者带来舒适的体验

项目地址： [点我传送](https://github.com/sml2h3/ddddocr) 

# 更新说明

  本次更新其实分为两部分，其中有一部分是在1.2.0版本就已经更新了，但是在这里还是有必要提一下的。

## 第一部分 OCR识别部分

  在1.2.0开始，ddddocr的识别部分进行了一次beta更新，主要更新在于网络结构主体的升级，其训练数据并没有发生过多的改变，所以理论上在识别结果上，原先可能识别效果的很好的图形在1.2.0上有一小部分概率会有一定程度的下降，也有可能原本识别不好的图形在1.2.0之后效果却变得特别好。
  测试代码：
   

```python
import ddddocr

ocr = ddddocr.DdddOcr()

with open("test.jpg", 'rb') as f:
    image = f.read()

res = ocr.classification(image)
print(res)
``` 
由于事实上确实在一些图片上老版本的模型识别效果比新模型好，特地这次更新把老模型也加入进去了，通过在初始化ddddocr的时候使用old参数即可快速切换老模型

```python
import ddddocr

ocr = ddddocr.DdddOcr(old=True)

with open("test.jpg", 'rb') as f:
    image = f.read()

res = ocr.classification(image)
print(res)
``` 

  OCR部分应该已经有很多人做了测试，在这里就放一部分网友的测试图片。

   ![Test](https://cdn.wenanzhe.com/img/20210715211733855.png) 
   ![Test](https://cdn.wenanzhe.com/img/78b7f57d-371d-4b65-afb2-d19608ae1892.png) 
  ![Test](https://cdn.wenanzhe.com/img/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20211226142305.png) 
   ![Test](https://cdn.wenanzhe.com/img/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20211226142325.png) 
   ![Test](https://cdn.wenanzhe.com/img/2AMLyA_fd83e1f1800e829033417ae6dd0e0ae0.png) 
   ![Test](https://cdn.wenanzhe.com/img/aabd_181ae81dd5526b8b89f987d1179266ce.jpg) 
   ![Test](https://cdn.wenanzhe.com/img/2bghz_b504e9f9de1ed7070102d21c6481e0cf.png) 
   ![Test](https://cdn.wenanzhe.com/img/0000_z4ecc2p65rxc610x.jpg) 
   ![Test](https://cdn.wenanzhe.com/img/2acd_0586b6b36858a4e8a9939db8a7ec07b7.jpg) 
  ![Test](https://cdn.wenanzhe.com/img/2a8r_79074e311d573d31e1630978fe04b990.jpg) 
   ![Test](https://cdn.wenanzhe.com/img/aftf_C2vHZlk8540y3qAmCM.bmp) 
   ![Test](https://cdn.wenanzhe.com/img/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20211226144057.png) 
等等更多图片等你测试哟~

## 第二部分 目标检测部分
  在本次1.3.0的更新中，目标检测部分隆重登场！
  目标检测部分同样也是由大量随机合成数据训练而成，对于现在已有的点选验证码图片或者未知的验证码图片都有可能具备一定的识别能力，适用于文字点选和图标点选。
  简单来说，对于点选类的验证码，可以快速的检测出图片上的文字或者图标。
  

```python
import ddddocr
import cv2

det = ddddocr.DdddOcr(det=True)

with open("test.jpg", 'rb') as f:
    image = f.read()

poses = det.detection(image)
print(poses)

im = cv2.imread("test.jpg")

for box in poses:
    x1, y1, x2, y2 = box
    im = cv2.rectangle(im, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

cv2.imwrite("result.jpg", im)

```

举些例子：

 ![Test](https://cdn.wenanzhe.com/img/page1_1.jpg) 
   ![Test](https://cdn.wenanzhe.com/img/page1_2.jpg) 
   ![Test](https://cdn.wenanzhe.com/img/page1_3.jpg) 
   ![Test](https://cdn.wenanzhe.com/img/page1_4.jpg) 
   ![Test](https://cdn.wenanzhe.com/img/result.jpg) 
  ![Test](https://cdn.wenanzhe.com/img/result2.jpg) 
  ![Test](https://cdn.wenanzhe.com/img/result4.jpg) 

以上只是目前我能找到的点选验证码图片，做了一个简单的测试。

# 安装

## 环境支持

`python <= 3.10`

`Windows/Linux/Macos..`

暂时不支持Macbook M1(X)，M1(X)用户需要自己编译onnxruntime才可以使用

## 安装命令

`pip install ddddocr`

以上命令将自动安装符合自己电脑环境的最新ddddocr

## 拓展 一键部署ddddocr api，支持docker部署

[github](https://github.com/sml2h3/ocr_api_server) 

[gitee](https://gitee.com/fkgeek/ocr_api_server)

## 爬虫框架推荐

[feapder](https://github.com/Boris-code/feapder) 

[crawlab](https://github.com/crawlab-team/crawlab)

# 交流群 （加我好友拉你进群）
 
 ![四群链接](https://cdn.wenanzhe.com/img/1d5ec7445d745a4d6790df352e1abb5.png!/scale/50)
 ![Test](https://cdn.wenanzhe.com/img/mmqrcode1640418911274.png!/scale/50) 


   

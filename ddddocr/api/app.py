#!/usr/bin/env python
# -*- coding: utf-8 -*-
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:  # pragma: no cover - fallback for older pydantic
    from pydantic import BaseModel, Field, validator as field_validator
import uvicorn
import base64
import io
import os
import binascii
from typing import Optional, List, Union, Dict, Any
import time
from PIL import Image
import logging
import sys
import json

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("ddddocr-api")

# 导入 ddddocr
try:
    from . import DdddOcr, DdddOcrInputError, InvalidImageError, MAX_IMAGE_BYTES as CORE_MAX_IMAGE_BYTES
except ImportError:
    import ddddocr
    DdddOcr = ddddocr.DdddOcr
    DdddOcrInputError = getattr(ddddocr, 'DdddOcrInputError', Exception)
    InvalidImageError = getattr(ddddocr, 'InvalidImageError', Exception)
    CORE_MAX_IMAGE_BYTES = getattr(ddddocr, 'MAX_IMAGE_BYTES', 8 * 1024 * 1024)

# 全局变量存储OCR实例
ocr_instances: Dict[str, Dict[str, Any]] = {}


def _validate_base64_payload(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 不能为空")
    try:
        decoded = base64.b64decode(value, validate=True)
    except binascii.Error as exc:
        raise ValueError(f"{field_name} 不是合法的Base64字符串") from exc
    if len(decoded) == 0:
        raise ValueError(f"{field_name} 内容为空")
    if len(decoded) > CORE_MAX_IMAGE_BYTES:
        raise ValueError(f"{field_name} 大小超过 {CORE_MAX_IMAGE_BYTES // 1024}KB 限制")
    return value


def _decode_base64_bytes(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except binascii.Error as exc:
        raise HTTPException(status_code=400, detail="Base64 内容错误") from exc


def _ensure_colors_list(data: List[Any]) -> List[str]:
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="colors 必须是字符串列表")
    normalized = []
    for item in data:
        if not isinstance(item, str):
            raise HTTPException(status_code=400, detail="colors 列表中必须是字符串")
        stripped = item.strip()
        if stripped:
            normalized.append(stripped)
    return normalized


def _validate_custom_range_dict(parsed: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
    if not isinstance(parsed, dict):
        raise ValueError("custom_color_ranges 必须是字典")
    for key, ranges in parsed.items():
        if not isinstance(key, str):
            raise ValueError("custom_color_ranges 的键必须为字符串")
        if not isinstance(ranges, list):
            raise ValueError("custom_color_ranges 的值必须为列表")
        for segment in ranges:
            if not isinstance(segment, list) or len(segment) != 3:
                raise ValueError("颜色区间必须是长度为3的列表")
            for value in segment:
                if not isinstance(value, int):
                    raise ValueError("颜色区间中的值需要为整数")
                if not 0 <= value <= 255:
                    raise ValueError("颜色区间的值需位于0-255之间")
    return parsed


def _ensure_custom_ranges(data: Any) -> Optional[Dict[str, List[List[int]]]]:
    if data in (None, "null", ""):
        return None
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="custom_color_ranges JSON 解析失败") from exc
    else:
        parsed = data
    if parsed is None:
        return None
    try:
        return _validate_custom_range_dict(parsed)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _coerce_bool_param(value: Union[bool, str], field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {'true', '1', 'yes', 'y'}:
            return True
        if lowered in {'false', '0', 'no', 'n'}:
            return False
    raise HTTPException(status_code=400, detail=f"{field_name} 只能是布尔值")

# 定义请求模型
class Base64Image(BaseModel):
    image: str = Field(..., description="Base64编码的图片数据")
    
    @field_validator('image')
    def validate_image(cls, value):
        return _validate_base64_payload(value, 'image')
    
class OCRRequest(Base64Image):
    probability: bool = Field(False, description="是否返回识别概率")
    colors: List[str] = Field(default_factory=list, description="颜色过滤列表")
    custom_color_ranges: Optional[Dict[str, List[List[int]]]] = Field(None, description="自定义颜色范围")

    @field_validator('colors')
    def validate_colors(cls, value):
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError('colors 必须是字符串列表')
        normalized = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError('colors 中的元素必须是字符串')
            stripped = item.strip()
            if not stripped:
                raise ValueError('colors 不允许包含空字符串')
            normalized.append(stripped)
        return normalized

    @field_validator('custom_color_ranges')
    def validate_custom_ranges(cls, value):
        if value is None:
            return value
        return _validate_custom_range_dict(value)
    
class SlideMatchRequest(BaseModel):
    target_image: str = Field(..., description="目标图片的Base64编码")
    background_image: str = Field(..., description="背景图片的Base64编码")
    simple_target: bool = Field(False, description="是否使用简化目标")
    flag: bool = Field(False, description="标记选项")

    @field_validator('target_image')
    def validate_target_image(cls, value):
        return _validate_base64_payload(value, 'target_image')

    @field_validator('background_image')
    def validate_background_image(cls, value):
        return _validate_base64_payload(value, 'background_image')
    
class SlideComparisonRequest(BaseModel):
    target_image: str = Field(..., description="目标图片的Base64编码")
    background_image: str = Field(..., description="背景图片的Base64编码")

    @field_validator('target_image')
    def validate_target_image(cls, value):
        return _validate_base64_payload(value, 'target_image')

    @field_validator('background_image')
    def validate_background_image(cls, value):
        return _validate_base64_payload(value, 'background_image')
    
class CharsetRangeRequest(BaseModel):
    charset_range: List[str] = Field(..., description="字符范围")

    @field_validator('charset_range')
    def validate_charset(cls, value):
        if not isinstance(value, list):
            raise ValueError('charset_range 需要为字符串列表')
        normalized = []
        for item in value:
            if not isinstance(item, str) or not item:
                raise ValueError('charset_range 需要为非空字符串')
            normalized.append(item)
        return normalized

class ModelConfig(BaseModel):
    ocr: bool = Field(True, description="是否启用OCR功能")
    det: bool = Field(False, description="是否启用目标检测功能")
    old: bool = Field(False, description="是否使用旧版OCR模型")
    beta: bool = Field(False, description="是否使用Beta版OCR模型")
    use_gpu: bool = Field(False, description="是否使用GPU加速")
    device_id: int = Field(0, description="GPU设备ID")
    show_ad: bool = Field(True, description="是否显示广告")
    import_onnx_path: str = Field("", description="自定义模型路径")
    charsets_path: str = Field("", description="自定义字符集路径")


class OCRResponse(BaseModel):
    result: Union[str, Dict[str, Any]]
    probability: Optional[Any] = None
    processing_time: float


class DetectionResponse(BaseModel):
    result: List[List[int]]
    processing_time: float


class SlideMatchResult(BaseModel):
    target_x: int
    target_y: int
    target: List[int]


class SlideMatchResponse(BaseModel):
    result: SlideMatchResult
    processing_time: float


class SlideComparisonResponse(BaseModel):
    result: Dict[str, List[int]]
    processing_time: float

# 函数：获取OCR实例
def get_ocr_instance(
    config_key: str,
    ocr: bool = True,
    det: bool = False,
    old: bool = False,
    beta: bool = False,
    use_gpu: bool = False,
    device_id: int = 0,
    show_ad: bool = True,
    import_onnx_path: str = "",
    charsets_path: str = ""
):
    """
    获取或创建OCR实例
    """
    if config_key in ocr_instances:
        ocr_instances[config_key]["last_used"] = time.time()
        return ocr_instances[config_key]["instance"]

    logger.info(f"创建新的OCR实例，配置: {config_key}")
    try:
        instance = DdddOcr(
            ocr=ocr,
            det=det,
            old=old,
            beta=beta,
            use_gpu=use_gpu,
            device_id=device_id,
            show_ad=show_ad,
            import_onnx_path=import_onnx_path,
            charsets_path=charsets_path
        )
        ocr_instances[config_key] = {
            "instance": instance,
            "last_used": time.time()
        }
        return instance
    except (DdddOcrInputError, InvalidImageError) as e:
        logger.error(f"创建OCR实例失败: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"创建OCR实例失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"初始化OCR失败: {str(e)}") from e

# 清理不活跃的OCR实例
def cleanup_inactive_instances(max_idle_time: int = 3600):
    """
    清理长时间不活跃的OCR实例以释放内存
    """
    global ocr_instances
    current_time = time.time()
    instances_to_remove = []
    
    for key, instance_data in ocr_instances.items():
        if current_time - instance_data["last_used"] > max_idle_time:
            instances_to_remove.append(key)
    
    for key in instances_to_remove:
        del ocr_instances[key]
        logger.info(f"已清理不活跃的OCR实例: {key}")

# 创建FastAPI应用
app = FastAPI(
    title="DdddOcr API",
    description="DdddOcr通用验证码识别API服务",
    version="1.6.1",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义API默认参数
default_ocr = os.environ.get("DDDDOCR_OCR", "true").lower() == "true"
default_det = os.environ.get("DDDDOCR_DET", "false").lower() == "true"
default_old = os.environ.get("DDDDOCR_OLD", "false").lower() == "true"
default_beta = os.environ.get("DDDDOCR_BETA", "false").lower() == "true"
default_use_gpu = os.environ.get("DDDDOCR_USE_GPU", "false").lower() == "true"
default_device_id = int(os.environ.get("DDDDOCR_DEVICE_ID", "0"))
default_show_ad = os.environ.get("DDDDOCR_SHOW_AD", "true").lower() == "true"
default_import_onnx_path = os.environ.get("DDDDOCR_IMPORT_ONNX_PATH", "")
default_charsets_path = os.environ.get("DDDDOCR_CHARSETS_PATH", "")

# 健康检查端点
@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}

# OCR识别端点 - JSON请求
@app.post("/ocr", response_model=OCRResponse)
async def ocr_recognition(
    request: OCRRequest,
    background_tasks: BackgroundTasks,
    ocr: bool = Query(default_ocr, description="是否启用OCR功能"),
    det: bool = Query(default_det, description="是否启用目标检测功能"),
    old: bool = Query(default_old, description="是否使用旧版OCR模型"),
    beta: bool = Query(default_beta, description="是否使用Beta版OCR模型"),
    use_gpu: bool = Query(default_use_gpu, description="是否使用GPU加速"),
    device_id: int = Query(default_device_id, description="GPU设备ID"),
    show_ad: bool = Query(default_show_ad, description="是否显示广告")
):
    """
    OCR文字识别 - 接收Base64编码的图片
    """
    image = None
    try:
        img_data = _decode_base64_bytes(request.image)
        image = Image.open(io.BytesIO(img_data))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"OCR请求图片解析失败: {str(exc)}")
        raise HTTPException(status_code=400, detail="无法读取图片") from exc

    config_key = f"ocr={ocr}-det={det}-old={old}-beta={beta}-gpu={use_gpu}-dev={device_id}"

    ocr_instance = get_ocr_instance(
        config_key, ocr, det, old, beta, use_gpu, device_id, show_ad,
        default_import_onnx_path, default_charsets_path
    )

    start_time = time.time()
    try:
        if request.probability:
            result = ocr_instance.classification(
                image,
                probability=True,
                color_filter_colors=request.colors,
                color_filter_custom_ranges=request.custom_color_ranges
            )
            response_data = {
                "result": result,
                "processing_time": time.time() - start_time
            }
        else:
            result = ocr_instance.classification(
                image,
                color_filter_colors=request.colors,
                color_filter_custom_ranges=request.custom_color_ranges
            )
            response_data = {
                "result": result,
                "processing_time": time.time() - start_time
            }
    except (DdddOcrInputError, InvalidImageError) as exc:
        logger.warning(f"OCR识别参数错误: {str(exc)}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"OCR识别失败: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"OCR识别失败: {str(exc)}") from exc
    finally:
        if image is not None:
            try:
                image.close()
            except Exception:
                pass

    background_tasks.add_task(cleanup_inactive_instances)
    return response_data

# OCR识别端点 - 文件上传
# OCR识别端点 - 文件上传
@app.post("/ocr/file", response_model=OCRResponse)
async def ocr_recognition_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    probability: Union[bool, str] = Form(False),
    colors: str = Form("[]"),
    custom_color_ranges: str = Form("null"),
    ocr: bool = Query(default_ocr, description="是否启用OCR功能"),
    det: bool = Query(default_det, description="是否启用目标检测功能"),
    old: bool = Query(default_old, description="是否使用旧版OCR模型"),
    beta: bool = Query(default_beta, description="是否使用Beta版OCR模型"),
    use_gpu: bool = Query(default_use_gpu, description="是否使用GPU加速"),
    device_id: int = Query(default_device_id, description="GPU设备ID"),
    show_ad: bool = Query(default_show_ad, description="是否显示广告")
):
    """
    OCR文字识别 - 接收上传的图片文件
    """
    image = None
    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="无法读取上传文件") from exc

    if not contents:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(contents) > CORE_MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"图片大小超过 {CORE_MAX_IMAGE_BYTES // 1024}KB 限制"
        )
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail="无法解析上传的图片") from exc

    try:
        colors_data = json.loads(colors) if colors else []
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="colors JSON 解析失败") from exc
    colors_list = _ensure_colors_list(colors_data)
    custom_ranges = _ensure_custom_ranges(custom_color_ranges)
    probability_flag = _coerce_bool_param(probability, 'probability')

    config_key = f"ocr={ocr}-det={det}-old={old}-beta={beta}-gpu={use_gpu}-dev={device_id}"

    ocr_instance = get_ocr_instance(
        config_key, ocr, det, old, beta, use_gpu, device_id, show_ad,
        default_import_onnx_path, default_charsets_path
    )

    start_time = time.time()
    try:
        if probability_flag:
            result = ocr_instance.classification(
                image,
                probability=True,
                color_filter_colors=colors_list,
                color_filter_custom_ranges=custom_ranges
            )
            response_data = {
                "result": result,
                "processing_time": time.time() - start_time
            }
        else:
            result = ocr_instance.classification(
                image,
                color_filter_colors=colors_list,
                color_filter_custom_ranges=custom_ranges
            )
            response_data = {
                "result": result,
                "processing_time": time.time() - start_time
            }
    except (DdddOcrInputError, InvalidImageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"OCR文件识别失败: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"OCR识别失败: {str(exc)}") from exc
    finally:
        if image is not None:
            try:
                image.close()
            except Exception:
                pass

    background_tasks.add_task(cleanup_inactive_instances)
    return response_data

# 目标检测端点
# 目标检测端点
# 目标检测端点
@app.post("/det", response_model=DetectionResponse)
async def object_detection(
    request: Base64Image,
    background_tasks: BackgroundTasks,
    ocr: bool = Query(False, description="是否启用OCR功能"),
    det: bool = Query(True, description="是否启用目标检测功能"),
    use_gpu: bool = Query(default_use_gpu, description="是否使用GPU加速"),
    device_id: int = Query(default_device_id, description="GPU设备ID"),
    show_ad: bool = Query(default_show_ad, description="是否显示广告")
):
    """
    目标检测功能 - 接收Base64编码的图片
    """
    try:
        img_data = _decode_base64_bytes(request.image)
    except HTTPException:
        raise

    config_key = f"ocr={ocr}-det={det}-gpu={use_gpu}-dev={device_id}"

    ocr_instance = get_ocr_instance(
        config_key, ocr, det, False, False, use_gpu, device_id, show_ad,
        default_import_onnx_path, default_charsets_path
    )

    start_time = time.time()
    try:
        result = ocr_instance.detection(img_bytes=img_data)
    except (DdddOcrInputError, InvalidImageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"目标检测失败: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"目标检测失败: {str(exc)}") from exc

    background_tasks.add_task(cleanup_inactive_instances)

    return {
        "result": result,
        "processing_time": time.time() - start_time
    }

# 目标检测端点 - 文件上传
@app.post("/det/file", response_model=DetectionResponse)
async def object_detection_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    ocr: bool = Query(False, description="是否启用OCR功能"),
    det: bool = Query(True, description="是否启用目标检测功能"),
    use_gpu: bool = Query(default_use_gpu, description="是否使用GPU加速"),
    device_id: int = Query(default_device_id, description="GPU设备ID"),
    show_ad: bool = Query(default_show_ad, description="是否显示广告")
):
    """
    目标检测功能 - 接收上传的图片文件
    """
    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="无法读取上传文件") from exc

    if not contents:
        raise HTTPException(status_code=400, detail="上传文件为空")
    if len(contents) > CORE_MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"图片大小超过 {CORE_MAX_IMAGE_BYTES // 1024}KB 限制"
        )

    config_key = f"ocr={ocr}-det={det}-gpu={use_gpu}-dev={device_id}"

    ocr_instance = get_ocr_instance(
        config_key, ocr, det, False, False, use_gpu, device_id, show_ad,
        default_import_onnx_path, default_charsets_path
    )

    start_time = time.time()
    try:
        result = ocr_instance.detection(img_bytes=contents)
    except (DdddOcrInputError, InvalidImageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"目标检测文件识别失败: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"目标检测失败: {str(exc)}") from exc

    background_tasks.add_task(cleanup_inactive_instances)

    return {
        "result": result,
        "processing_time": time.time() - start_time
    }

# 滑块匹配端点
@app.post("/slide_match", response_model=SlideMatchResponse)
async def slide_match_recognition(
    request: SlideMatchRequest,
    background_tasks: BackgroundTasks,
    ocr: bool = Query(False, description="是否启用OCR功能"),
    det: bool = Query(False, description="是否启用目标检测功能"),
    use_gpu: bool = Query(default_use_gpu, description="是否使用GPU加速"),
    device_id: int = Query(default_device_id, description="GPU设备ID"),
    show_ad: bool = Query(default_show_ad, description="是否显示广告")
):
    """
    滑块验证码匹配 - 接收Base64编码的目标图和背景图
    """
    target_data = _decode_base64_bytes(request.target_image)
    background_data = _decode_base64_bytes(request.background_image)

    config_key = f"ocr={ocr}-det={det}-gpu={use_gpu}-dev={device_id}"

    ocr_instance = get_ocr_instance(
        config_key, ocr, det, False, False, use_gpu, device_id, show_ad,
        default_import_onnx_path, default_charsets_path
    )

    start_time = time.time()
    try:
        result = ocr_instance.slide_match(
            target_bytes=target_data,
            background_bytes=background_data,
            simple_target=request.simple_target,
            flag=request.flag
        )
    except (DdddOcrInputError, InvalidImageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"滑块匹配失败: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"滑块匹配失败: {str(exc)}") from exc

    background_tasks.add_task(cleanup_inactive_instances)
    return {
        "result": result,
        "processing_time": time.time() - start_time
    }

# 滑块比较端点
@app.post("/slide_comparison", response_model=SlideComparisonResponse)
async def slide_comparison_recognition(
    request: SlideComparisonRequest,
    background_tasks: BackgroundTasks,
    ocr: bool = Query(False, description="是否启用OCR功能"),
    det: bool = Query(False, description="是否启用目标检测功能"),
    use_gpu: bool = Query(default_use_gpu, description="是否使用GPU加速"),
    device_id: int = Query(default_device_id, description="GPU设备ID"),
    show_ad: bool = Query(default_show_ad, description="是否显示广告")
):
    """
    滑块验证码图像差异比较 - 接收Base64编码的目标图和背景图
    """
    target_data = _decode_base64_bytes(request.target_image)
    background_data = _decode_base64_bytes(request.background_image)

    config_key = f"ocr={ocr}-det={det}-gpu={use_gpu}-dev={device_id}"

    ocr_instance = get_ocr_instance(
        config_key, ocr, det, False, False, use_gpu, device_id, show_ad,
        default_import_onnx_path, default_charsets_path
    )

    start_time = time.time()
    try:
        result = ocr_instance.slide_comparison(
            target_bytes=target_data,
            background_bytes=background_data
        )
    except (DdddOcrInputError, InvalidImageError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"滑块比较失败: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"滑块比较失败: {str(exc)}") from exc

    background_tasks.add_task(cleanup_inactive_instances)

    return {
        "result": result,
        "processing_time": time.time() - start_time
    }

# 设置字符范围端点
# 设置字符范围端点
@app.post("/set_charset_range")
async def set_charset_range(
    request: CharsetRangeRequest,
    background_tasks: BackgroundTasks,
    ocr: bool = Query(True, description="是否启用OCR功能"),
    det: bool = Query(False, description="是否启用目标检测功能"),
    old: bool = Query(default_old, description="是否使用旧版OCR模型"),
    beta: bool = Query(default_beta, description="是否使用Beta版OCR模型"),
    use_gpu: bool = Query(default_use_gpu, description="是否使用GPU加速"),
    device_id: int = Query(default_device_id, description="GPU设备ID"),
    show_ad: bool = Query(default_show_ad, description="是否显示广告")
):
    """
    设置OCR识别的字符范围
    """
    try:
        config_key = f"ocr={ocr}-det={det}-old={old}-beta={beta}-gpu={use_gpu}-dev={device_id}"

        ocr_instance = get_ocr_instance(
            config_key, ocr, det, old, beta, use_gpu, device_id, show_ad,
            default_import_onnx_path, default_charsets_path
        )

        start_time = time.time()
        ocr_instance.set_ranges(request.charset_range)

        background_tasks.add_task(cleanup_inactive_instances)

        return {
            "result": "字符范围设置成功",
            "charset_range": request.charset_range,
            "processing_time": time.time() - start_time
        }

    except (DdddOcrInputError, InvalidImageError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as e:
        logger.error(f"设置字符范围失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"设置字符范围失败: {str(e)}") from e

# 获取当前配置信息
@app.get("/config")
async def get_current_config():
    return {
        "default_config": {
            "ocr": default_ocr,
            "det": default_det,
            "old": default_old,
            "beta": default_beta,
            "use_gpu": default_use_gpu,
            "device_id": default_device_id,
            "show_ad": default_show_ad,
            "import_onnx_path": default_import_onnx_path,
            "charsets_path": default_charsets_path
        },
        "active_instances": len(ocr_instances),
        "environment": {
            "python_version": sys.version,
            "time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    }

# 主函数，用于直接运行此文件时启动服务
def main():
    # 获取环境变量或使用默认值
    host = os.environ.get("DDDDOCR_HOST", "127.0.0.1")
    port = int(os.environ.get("DDDDOCR_PORT", "8000"))
    workers = int(os.environ.get("DDDDOCR_WORKERS", "1"))
    
    # 启动服务
    print(f"启动DdddOcr API服务在 {host}:{port}，工作进程数: {workers}")
    uvicorn.run("ddddocr.api:app", host=host, port=port, workers=workers)

if __name__ == "__main__":
    main() 

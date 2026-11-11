# eye_gesture_control

基于 **dlib 68 点关键点 + RetinaFace + ArcFace + YOLO-ONNX 手势模型** 的多模态识别程序：摄像头采集后做眼动追踪 / 手势识别 / 人脸识别，把识别结果通过**串口**下发给 Arduino 小车。

本仓库由原 `mul_pro` 工程整理而来，只保留三个主脚本及其真正被 import / 读取的代码、模型与人脸库，另外附上 Arduino 固件与键盘串口测试工具。

## 功能概览

| 脚本 | 做什么 | 关键依赖 |
| --- | --- | --- |
| `main_control.py` | dlib 人脸检测 + 68 点关键点，取 36–47 号眼部点做瞳孔检测（CLAHE、自适应阈值、轮廓圆度筛选），加权中值 + 卡尔曼平滑，动态阈值判断眼球方向并下发串口指令 | `shape_predictor_68_face_landmarks.dat` |
| `detect_all.py` | 手势 + 眼动融合控制（人脸识别模块默认被注释）。手势：`best.onnx` 推理 + NMS + 类别稳定；眼动：dlib EAR 判断闭眼、瞳孔相对位置校准、方向映射 | `best.onnx`、`retinaface.py`、`models/`、`utils/facealign.py`、`checkpoints/resnet18_110.pth`、`test/` 人脸库 |
| `retinaface.py` | RetinaFace(MobileNet0.25) 检测封装，`detect_image()` 返回人脸框与 5 点关键点，供 `detect_all.py` 做人脸对齐 | `retinaface_arcface/nets/`、`retinaface_arcface/utils/`、`model_data/Retinaface_mobilenet0.25.pth` |

调用关系：

```
main_control.py ──→ shape_predictor_68_face_landmarks.dat
detect_all.py ──→ best.onnx（手势）
              ──→ retinaface.py ──→ retinaface_arcface/nets/* + retinaface_arcface/utils/* + model_data/*.pth
              ──→ models/*（ArcFace resnet18）+ checkpoints/resnet18_110.pth（人脸特征）
              ──→ utils/facealign.py（人脸对齐 + 余弦相似度）
              ──→ test/<人名>/<照片>（人脸库）
```

## 目录结构

```
eye_gesture_control/
├── main_control.py                 # 眼动控制（dlib）
├── detect_all.py                   # 手势 + 眼动融合控制（主程序）
├── retinaface.py                   # RetinaFace 检测封装
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── run_main_control.bat            # 双击启动 main_control.py
├── run_detect_all.bat              # 双击启动 detect_all.py
├── best.onnx                       # 手势识别模型
├── shape_predictor_68_face_landmarks.dat   # dlib 68 点关键点模型
├── model_data/
│   └── Retinaface_mobilenet0.25.pth        # RetinaFace 权重
├── checkpoints/
│   └── resnet18_110.pth                    # ArcFace resnet18 权重
├── models/                         # ArcFace 主干网络（resnet18/34/50）
│   ├── __init__.py
│   ├── resnet.py
│   ├── metrics.py
│   └── focal_loss.py
├── utils/
│   └── facealign.py                # 人脸对齐、特征提取、余弦相似度、人脸库加载
├── retinaface_arcface/             # RetinaFace 网络与工具
│   ├── nets/
│   │   ├── layers.py
│   │   ├── mobilenet025.py
│   │   └── retinaface.py
│   ├── utils/
│   │   ├── anchors.py
│   │   ├── box_utils.py
│   │   └── config.py
│   └── models/                     # utils/facealign.py 会 import 它
│       ├── __init__.py
│       ├── resnet.py
│       ├── metrics.py
│       └── focal_loss.py
├── test/                           # 人脸库：一人一个文件夹
│   ├── ADMIN/…
│   ├── Hujie/…
│   ├── HYY/…
│   └── YX/…
└── hardware/
    └── arduino_demo/
        ├── ardino_demo.ino         # Arduino 固件（9600 波特率）
        └── serial_wsad/
            ├── keyboard_serial.py  # 键盘 → 串口测试（配 ardino_demo.ino）
            ├── serial_wsad.ino     # 另一套 W/S/A/D 协议固件
            ├── start_keyboard.bat
            ├── README.md
            └── requirements.txt
```

## 环境

- Windows 10/11（脚本用到 `cv2.imshow` 窗口、全局键盘钩子）
- Python 3.10 —— 本机环境 `D:\Anaconda_envs\mul_pro`（Python 3.10.22），`requirements.txt` 中的版本号取自该环境实际安装版本
- 可选：NVIDIA GPU + CUDA，有则 RetinaFace/ArcFace 自动用 GPU，onnxruntime 优先 CUDA，失败自动回退 CPU

```powershell
& 'D:\Anaconda_envs\mul_pro\python.exe' -m pip install -r requirements.txt
```

## 运行

> **必须在项目根目录运行**：代码里全是相对路径（`model_data/`、`checkpoints/`、`test/`、`best.onnx`、`retinaface_arcface/`），换目录会直接报找不到文件。

```powershell
cd C:\Users\Guards\Desktop\eye_gesture_control

# 眼动控制
& 'D:\Anaconda_envs\mul_pro\python.exe' main_control.py

# 手势 + 眼动融合控制
& 'D:\Anaconda_envs\mul_pro\python.exe' detect_all.py `
    --pic-dir test/ `
    --gesture-model best.onnx `
    --landmark-model shape_predictor_68_face_landmarks.dat `
    --serial-port COM5
```

也可以直接双击 `run_main_control.bat` / `run_detect_all.bat`（内部固定使用 `D:\Anaconda_envs\mul_pro\python.exe`，并会先 `cd` 到脚本所在目录）。

`detect_all.py` 参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--gesture-model` | `best.onnx` | 手势 ONNX 模型路径 |
| `--pic-dir` | `test/` | 人脸库文件夹 |
| `--video` | `0` | 摄像头编号 |
| `--distance` | `0.5` | 人脸识别余弦相似度阈值 |
| `--landmark-model` | `shape_predictor_68_face_landmarks.dat` | dlib 关键点模型 |
| `--serial-port` | `COM5` | 串口号 |
| `--baudrate` | `9600` | 波特率 |
| `--face-verify-interval` | `30.0` | 人脸验证有效期（秒） |
| `--show-face` / `--show-gesture` / `--show-eye` | `True` | 是否显示对应识别结果 |

> 注意：三个 `--show-*` 参数用的是 `type=bool`，命令行写 `--show-eye False` 得到的仍是 `True`（`bool("False") == True`）。要真正关掉某个模块，得改代码或直接不传该分支，别指望命令行关闭。

按 `q` 退出；`main_control.py` 的串口固定写在 `__main__` 里（`COM5`，9600）。

## 串口协议

PC → Arduino，均为**单字节 + `\n`**，9600 8N1。

| 发送方 | 指令 | 含义 |
| --- | --- | --- |
| `main_control.py` | `0\n` | 停止 |
| | `1\n` | 前进 |
| | `2\n` | 后退 |
| | `3\n` | 左转 |
| | `4\n` | 右转 |
| `detect_all.py`（手势） | `0\n`–`4\n` | 手势类别直接映射，类别 5 不发指令 |
| `detect_all.py`（眼动） | `0\n`–`4\n` | 注视→1，左看→3，右看→4，上看→1，下看→2，闭眼→0 |

`hardware/arduino_demo/ardino_demo.ino` 实际解析的是：`1`=M1 前进 1 秒、`2`=M2 前进 1 秒、`3`=M1 后退 1 秒、`4`=M2 后退 1 秒、`F`=两路前进 2 秒、`S`=停止，其它字符忽略。

> ⚠️ **已知不一致（本次只记录、未改动代码）**：
> 1. Python 端的 `0`（停止）在 `ardino_demo.ino` 里没有对应分支，属无效字符；实际停止要用 `S`。
> 2. 两侧 `1/2/3/4` 语义不同：Python 是整车前进/后退/左转/右转，固件是单电机动作。联调前必须先把协议对齐。
> 3. `serial_wsad/serial_wsad.ino` 是另一套 W/S/A/D 协议，与上述 Python 端均不匹配，仅作独立测试用。

## 模型文件清单

| 文件 | 大小 | 用途 | 是否纳入 git |
| --- | --- | --- | --- |
| `best.onnx` | 27.2 MB | 手势识别（`detect_all.py`） | 是 |
| `model_data/Retinaface_mobilenet0.25.pth` | 1.7 MB | RetinaFace 人脸检测（`retinaface.py`） | 是 |
| `checkpoints/resnet18_110.pth` | 97.8 MB | ArcFace resnet18 人脸特征（`detect_all.py`） | **否**（接近/超过 GitHub 单文件 100 MB 硬限制） |
| `shape_predictor_68_face_landmarks.dat` | 95.1 MB | dlib 68 点关键点（`main_control.py`、`detect_all.py`） | **否**（体积过大，建议单独下载） |

上一行两个“否”的文件**已在本地文件夹内**（可直接运行），只是被 `.gitignore` 排除。重新克隆仓库后需自行放回同名相对位置：

- dlib 模型：从 <http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2> 解压得到
- ArcFace 权重：取自原工程 `mul_pro/checkpoints/resnet18_110.pth`，公开来源不稳定，需自行留存

如需把大文件也入库，请改用 Git LFS：

```powershell
git lfs install
git lfs track "*.pth" "*.dat"
git add .gitattributes
```

## 上传到 GitHub

本目录已执行 `git init` 并完成首次提交，远程仓库需你自己建好后关联：

```powershell
cd C:\Users\Guards\Desktop\eye_gesture_control
git remote add origin https://github.com/<你的账号>/eye_gesture_control.git
git branch -M main
git push -u origin main
```

推送前确认 `git ls-files` 里没有超过 100 MB 的文件（`checkpoints/resnet18_110.pth`、`shape_predictor_68_face_landmarks.dat` 已被忽略）。

## 使用注意

- 首次运行需要静止校准：`main_control.py` 采集 60 帧，`detect_all.py` 眼动采集 30 帧，期间保持头部不动、直视前方。
- 串口不存在时程序不会退出，只是把指令丢弃（`main_control.py` 打印“串口连接失败”，`detect_all.py` 打印“串口未连接，指令被丢弃”）。
- `detect_all.py` 里的人脸验证流程（`process_face_recognition`）默认被注释掉了，当前只跑手势 + 眼动；`test/` 人脸库仍会在启动时加载。
- 原工程遗留的 `sys.path.append('/retinaface_arcface')`、`sys.path.append('e:/retinaface_arcface')` 是历史残留，不生效但无害，本次未改动业务代码。
- 摄像头帧率低或分辨率过高会明显掉帧，可在代码里调整 `CAP_PROP_FRAME_WIDTH/HEIGHT`。
- `test/` 下是**真实人脸照片**（按人名分文件夹）。推送到公开仓库前请确认你有这些照片的发布授权，或换成自己的测试图。

## 整理说明（来源与取舍）

- 代码来源：`C:\Users\Guards\Desktop\contact\code\code\mul_pro3\mul_pro`
- 硬件来源：`C:\Users\Guards\Desktop\contact\ardino_demo`
- 收录原则：只收三个主脚本真正 import / 读取的东西，外加指定要整理的 `ardino_demo`。
- **未收录**：`New/`（去反光）、`EyeControl_main/`（眨眼鼠标控制）、`iTracker/`（凝视模型）、`_backup_20261015-170700/`、`retinaface_arcface/` 下的其它 demo（`onnxtest.py`、`predict.py`、`test_all.py`、`test2_all.py`、`test.py`、`server.py`、`eye_tracker.py`）、`best.pt`、`mobilenetV1X0.25_pretrain.tar`、根目录 `utils/{anchors,box_utils,config}.py`（无脚本 import，实际用的是 `retinaface_arcface/utils/` 下的同名文件）、`__pycache__`、`.idea`。
- 原 `Readme.md` 只有三行说明（手势→`onnxtest.py`、人脸→`predict.py`、融合→`detect_all.py`），已并入本文；三个被点名的脚本只保留了 `detect_all.py`（`onnxtest.py`、`predict.py` 属未收录的独立 demo）。
- 原始目录未做任何修改。

## 许可

MIT，见 [LICENSE](LICENSE)。请把其中的版权人替换成你自己。

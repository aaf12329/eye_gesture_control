# 原始 Arduino 程序的键盘串口测试

keyboard_serial.py 当前匹配上一级的 ardino_demo.ino，波特率 9600。
原始 Arduino 源文件未修改。本目录的 serial_wsad.ino 是另一套 WSAD 协议，当前 Python 脚本不与它配套。

## 使用

1. 开发板烧录上一级的 ardino_demo.ino，随后关闭 Arduino 串口监视器。
2. 双击 start_keyboard.bat，选择开发板串口（最近烧录使用 COM5，以实际连接为准）。
3. 实时检测以下按键，无需按回车；支持大小写。

| 按键 | 发送的单字节 ASCII | 动作 |
| --- | --- | --- |
| 1 | 1 | M1 前进 1 秒 |
| 2 | 2 | M2 前进 1 秒 |
| 3 | 3 | M1 后退 1 秒 |
| 4 | 4 | M2 后退 1 秒 |
| F | F | 两路前进 1 秒 |
| S、X、空格、松开全部动作键 | S | 请求停止 |
| Esc / Ctrl+C | S，然后退出 | 请求停止并退出 |

同时按多个动作键时发送 S。只有指令变化时才发送；按住不会重复排队，松开再按可重新测试。
默认每 10 ms 检测一次键盘，每 100 ms 持续打印当前检测状态；按键变化时立即打印。
例如按住 1 会持续显示 [KEY] 1 -> 1  M1前进，松开显示 [KEY] 无按键 -> S  停止。
[TX] 表示已写入电脑串口，[KEY] 表示当前按键对应的指令，均不代表电机实际状态。

原 Arduino 程序每次动作有 delay(1000)，期间无法处理新指令，包括停止指令。
所以松键虽立即发送 S，开发板仍要等当前这 1 秒动作结束后处理。
原程序没有两路同时后退或原地左右转指令，因此测试键同步改为 1/2/3/4/F/S。

键盘使用 Windows 全局检测，切换窗口后仍会响应。暂停控制请按 Esc 退出。

## 环境和命令

启动脚本固定使用 D:\Anaconda_envs\mul_pro\python.exe，不创建新环境。
该环境已安装 pyserial。如以后缺失，可在 PowerShell 执行：

```powershell
& 'D:\Anaconda_envs\mul_pro\python.exe' -m pip install -r 'C:\Users\Guards\Desktop\eye_gesture_control\hardware\arduino_demo\serial_wsad\requirements.txt'
```

指定串口运行：

```powershell
& 'D:\Anaconda_envs\mul_pro\python.exe' -u 'C:\Users\Guards\Desktop\eye_gesture_control\hardware\arduino_demo\serial_wsad\keyboard_serial.py' --port COM5
```

加 --list-ports 只列出串口，不连接或发送指令。
原程序电机引脚：M1 为 4/6/9，M2 为 5/7/10，PWM 为 40。

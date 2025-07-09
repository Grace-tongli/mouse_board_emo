"""
主启动脚本：同时运行键盘、情绪和鼠标监控系统
支持按ESC键停止整个系统
"""

import threading
import time
import logging
from src.monitoring.keyboard_monitor import KeyboardMonitor
from src.monitoring.emotion_monitor import EmotionMonitor
from src.monitoring.mouse_monitor import MouseMonitor  # 导入鼠标监控器

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../../logs/monitor_system.log"),
        logging.StreamHandler()
    ]
)

def run_monitors():
    """运行整个监控系统"""
    logging.info("编程行为监控系统启动")

    # 创建共享的停止事件
    stop_event = threading.Event()

    try:
        # 创建监控器实例
        kb_monitor = KeyboardMonitor(
            analysis_interval=120,
            output_dir="../../data/raw/keyboard_data",
            stop_event=stop_event
        )

        em_monitor = EmotionMonitor(
            interval=120,
            output_file="../../data/raw/emotion_responses.csv",
            stop_event=stop_event
        )

        mouse_monitor = MouseMonitor(
            analysis_interval=120,
            output_dir="../../data/raw/mouse_data",
            stop_event=stop_event
        )

        # 创建并启动线程
        kb_thread = threading.Thread(target=kb_monitor.run)
        em_thread = threading.Thread(target=em_monitor.run)
        mouse_thread = threading.Thread(target=mouse_monitor.run)

        kb_thread.daemon = True
        em_thread.daemon = True
        mouse_thread.daemon = True

        kb_thread.start()
        em_thread.start()
        mouse_thread.start()

        logging.info("监控系统已启动，按ESC键停止...")

        # 主循环 - 等待停止事件
        while not stop_event.is_set():
            time.sleep(1)

        # 停止监控器
        kb_monitor.stop_listener()
        em_monitor.stop()
        mouse_monitor.stop_listener()

        # 等待线程结束
        kb_thread.join(timeout=2)
        em_thread.join(timeout=2)
        mouse_thread.join(timeout=2)

    except KeyboardInterrupt:
        logging.info("系统被用户中断")
        stop_event.set()
    except Exception as e:
        logging.error(f"系统发生错误: {str(e)}")
        stop_event.set()
    finally:
        logging.info("编程行为监控系统已停止")

if __name__ == "__main__":
    run_monitors()
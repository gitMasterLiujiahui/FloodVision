#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FloodRisk Web应用启动脚本
"""

import os
import sys
import time
import webbrowser
import socket


def _is_port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) != 0

def start_web_server():
    """启动Web服务器 (FastAPI + uvicorn)"""
    # 尝试不同的端口以避免冲突
    ports = [
        int(os.environ.get('PORT', 5000)),
        5001, 5002, 5003, 5004
    ]

    # 延迟打开浏览器的秒数
    open_delay_sec = 2

    for port in ports:
        if not _is_port_free(port):
            print(f"端口 {port} 被占用，尝试下一个端口")
            continue

        try:
            print(f"尝试在端口 {port} 启动服务器...")
            print(f"服务器将在 http://localhost:{port} 启动")

            # 启动uvicorn（以模块:app 形式）
            import uvicorn

            # 打开浏览器
            def _open_browser_later(url):
                time.sleep(open_delay_sec)
                try:
                    webbrowser.open(url)
                except Exception:
                    pass

            from threading import Thread
            Thread(target=_open_browser_later, args=(f'http://localhost:{port}',), daemon=True).start()

            # 运行服务（阻塞直到退出）
            uvicorn.run("web_app:app", host='0.0.0.0', port=port, reload=False)
            return True

        except Exception as e:
            print(f"启动失败: {e}")
            continue

    print("所有端口都被占用，请关闭其他Web服务器后重试")
    return False

def main():
    """主函数"""
    print("=" * 60)
    print("积水识别和车辆淹没部位判别系统 - Web版本")
    print("=" * 60)
    
    
    # 检查模型文件
    print("正在检查模型文件...")
    model_files = [
        "models/vehicle_detection/ssd.pt",
        "models/vehicle_detection/yolov11.pt", 
        "models/water_segmentation/deeplabv3.pth",
        "models/water_segmentation/yolov11.pt"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✓ {model_file}")
        else:
            print(f"⚠ {model_file} (文件不存在，部分功能可能受限)")
    
    print("\n正在启动Web应用...")
    
    print("按 Ctrl+C 停止服务器")
    
    try:
        if not start_web_server():
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n服务器已停止")

if __name__ == '__main__':
    main()
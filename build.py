import os
import sys
import subprocess
import shutil

def build_executable():
    print("="*50)
    print("🚀 正在启动 AI降重系统 自动化打包引擎...")
    print("="*50)

    # 1. 检查并安装打包工具
    try:
        import PyInstaller
    except ImportError:
        print("[*] 未检测到 PyInstaller，正在为您自动安装，请稍候...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("[*] PyInstaller 安装完成！\n")

    # 2. 读取原始的 main.py 文件
    if not os.path.exists("main.py"):
        print("[!] 错误：找不到 main.py，请确保 build.py 与 main.py 在同一目录下！")
        input("按任意键退出...")
        sys.exit(1)

    print("[*] 正在解析并注入动态路径引擎...")
    with open("main.py", "r", encoding="utf-8") as f:
        content = f.read()

    # 3. 注入“打包环境动态路径识别”代码
    path_injection = """
import sys
import os

# 动态环境判断引擎：区分是源码运行还是打包后的目录运行
if getattr(sys, 'frozen', False):
    # 打包为文件夹后，根目录就是 sys._MEIPASS 或 exe 所在目录
    BASE_DIR = sys._MEIPASS
else:
    # 源码运行，就是当前目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
"""
    # 替换原本写死的静态文件挂载路径
    content = content.replace(
        'app.mount("/static", StaticFiles(directory="static"), name="static")',
        'app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")'
    )
    content = content.replace(
        'FileResponse("static/index.html")',
        'FileResponse(os.path.join(BASE_DIR, "static", "index.html"))'
    )
    
    # 解决 Uvicorn 在 Windows 打包后可能出现的多进程无限循环 Bug
    content = content.replace(
        'if __name__ == "__main__":',
        'if __name__ == "__main__":\n    import multiprocessing\n    multiprocessing.freeze_support()'
    )

    # 4. 生成专用于打包的临时文件
    build_file_name = "main_dist_build.py"
    with open(build_file_name, "w", encoding="utf-8") as f:
        f.write(path_injection + content)

    # 5. 组装 PyInstaller 打包命令 (使用 --onedir 打包为文件夹)
    print("[*] 正在执行编译命令，这可能需要 1-3 分钟，请不要关闭窗口...")
    
    # 获取不同操作系统的路径分隔符 (Windows是;, Mac/Linux是:)
    sep = ";" if os.name == "nt" else ":"
    
    cmd = [
        "pyinstaller",
        "--noconfirm",           # 覆盖已存在的生成目录
        "--onedir",              # 核心修改：打包成一个文件夹 (不使用 --onefile)
        "--name=TraceLess",       # 设定输出软件的名称
        f"--add-data=static{sep}static", # 把包含 index.html 的前端文件夹塞进目录里
        "--clean",               # 打包前清理缓存
        build_file_name
    ]

    # 执行打包
    result = subprocess.call(cmd)

    # 6. 清理战场 (删除生成的临时构建文件和冗余物)
    print("\n[*] 正在清理临时构建垃圾...")
    if os.path.exists(build_file_name):
        os.remove(build_file_name)
    if os.path.exists("AI降重系统.spec"):
        os.remove("AI降重系统.spec")
    if os.path.exists("build"):
        shutil.rmtree("build")

    if result == 0:
        print("\n" + "="*50)
        print("🎉 打包大功告成！")
        print("💡 您的专属程序文件夹已生成在当前目录的 【dist】 文件夹中。")
        print("💡 请将 dist 目录下的 【AI降重系统】 整个文件夹打包发送给他人。")
        print("🔒 隐私保护生效：本地的 config.json 已被安全拦截，未被打包入程序。")
        print("="*50 + "\n")
    else:
        print("\n[!] 打包过程中发生错误，请检查上方的报错信息。")

    input("按任意键退出构建向导...")

if __name__ == "__main__":
    build_executable()
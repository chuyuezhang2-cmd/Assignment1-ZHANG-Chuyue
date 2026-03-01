import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("Welcome to Dynamic Maze Explorer!")
    print("1: Training model (run train.py)")
    print("2: Playing game (run play.py)")
    print("3: Exit")
    choice = input("Your choice: ")

    if choice == '1':
        from train import train
        train()
    elif choice == '2':
        try:
            from play import play, main_menu
            print("启动游戏菜单...")
            mode = main_menu()
            if mode in ["ai", "manual"]:
                play(mode)  # 传入模式参数
            else:
                print("无效模式选择，返回主菜单。")
                main()  # 如果菜单异常，返回主菜单
        except FileNotFoundError as e:
            print(f"错误：模型文件未找到 - {e}")
            print("请检查 src 目录下是否有 trained_model_ep800.pth 或类似文件。")
        except Exception as e:
            print(f"运行 play.py 时出错：{e}")
    elif choice == '3':
        sys.exit()
    else:
        print("无效选择，再试。")
        main()

if __name__ == "__main__":
    main()
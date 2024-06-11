import pyautogui
import os
import pywinctl as pwc
import subprocess
import json



# Selecting Window
subprocess.run(["open", "-a", "Obsidian"])
window = pwc.getWindowsWithTitle("Obsidian", condition=pwc.Re.CONTAINS)
menu = window[0].menu.getMenu()
print(json.dumps(menu, indent=4, ensure_ascii=False))

windows = pwc.getAllWindows()
pwc.getAllTitles()
pwc.getAllAppsNames()
pwc.getAllAppsWindowsTitles()
pwc.getAppsWithName("Obsidian")


def focus_on_obsidian():
    script = '''
    tell application "System Events"
        tell process "Obsidian"
            set frontmost to true
        end tell
    end tell
    '''
    os.system(f"osascript -e '{script}'")


# def open_obsidian_and_use_shortcut():
#     script = """
#     tell application "Obsidian"
#         activate
#     end tell
#     tell application "System Events"
#         tell process "Obsidian"
#             set frontmost to true
#             keystroke "o" using {command down}
#         end tell
#     end tell
#     """
#     os.system(f"osascript -e '{script}'")


# open_obsidian_and_use_shortcut()


# Usar a função para focar no Obsidian antes de executar ações com pyautogui
focus_on_obsidian()
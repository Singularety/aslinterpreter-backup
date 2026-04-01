; Outline: buildinstaller file used by the ASL interpreter project.
; Keep these settings/scripts in sync with app runtime expectations.

pip install pyinstaller
pyinstaller --onefile --windowed app/main.py
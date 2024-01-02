import os

xlsx_files = [f for f in os.listdir() if f.endswith('.xls')]
print(xlsx_files)
new_files = [(file, file[15:]) for file in xlsx_files]
print(new_files)

for original, new in new_files:
    os.rename(original, new)
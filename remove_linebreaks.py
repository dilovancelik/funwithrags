import os
import glob

os.chdir("data")
for file in glob.glob("*.txt"):
    with open(file) as f:
        current_text = f.read()
    with open(file, 'w') as f:
        new_text = ' '.join(current_text.split())
        f.write(new_text)

import os
from shlex import quote
import subprocess

prompts = [
    "settings1.txt",
    "settings2.txt",
]

for prompt in prompts:
    print("\n" + prompt)
    os.system("python disco.py \"" + prompt + "\"")

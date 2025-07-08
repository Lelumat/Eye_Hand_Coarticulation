import os
import sys
import subprocess

#Run the Classification.py script

for nComponents in range(1,10):
    print(f"Running Classification.py with {nComponents} components")
    subprocess.run(["python3", "Classification.py", str(nComponents)])
    print(f"Finished running Classification.py with {nComponents} components")
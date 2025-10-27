from nbformat import read, write, NotebookNode

# Define file paths
file_paths = [
    "data_analysis.ipynb",
    "pre_processing.ipynb",
    "part3.ipynb"
]

# Combine all notebooks into one
combined_notebook = NotebookNode({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5})

# Read and append all cells from each notebook
for path in file_paths:
    with open(path, "r", encoding="utf-8") as f:
        nb = read(f, as_version=4)
        combined_notebook["cells"].extend(nb["cells"])

# Save the combined notebook
output_path = "combined_notebook.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    write(combined_notebook, f)

output_path

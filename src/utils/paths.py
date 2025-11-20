import os

def get_project_root():
    return os.path.dirname(os.path.abspath(os.path.dirname("__file__")))

def get_data_path(filename: str) -> str:
    return os.path.join(get_project_root(), "notebooks", "data", filename)

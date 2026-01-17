from config import config

def print_all_paths():
    print("Available paths in config:\n")
    print("-" * 50)
    
    paths = config.get_all_paths()
    for key, path in paths.items():
        print(f"{key:15} : {path}")
    
    print("-" * 50)

if __name__ == "__main__":
    print_all_paths()

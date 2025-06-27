import os

# ✅ Use the actual path to your folder
directory = r"E:\MyProjects\AASA IT SOLUTION\Knuckles Tree\Official_Rep\Knuckles_Project\pramuka\FastRCNN1.9\test2"
count = 1

# List all jpg files in the directory
jpg_files = [f for f in os.listdir(directory) if f.lower().endswith(".jpg")]

# Rename files sequentially
for file in jpg_files:
    old_path = os.path.join(directory, file)
    new_name = f"test{count}.jpg"
    new_path = os.path.join(directory, new_name)

    try:
        os.rename(old_path, new_path)
        print(f"Renamed: {file} -> {new_name}")
        count += 1
    except Exception as e:
        print(f"Failed to rename {file}: {e}")

print("All files renamed successfully!")

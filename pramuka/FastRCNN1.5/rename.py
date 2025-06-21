import os

# Set the directory containing the .jpg files
directory = r"E:\MyProjects\AASA IT SOLUTION\Knuckles Tree\FastRCNN1.5\test"
count = 1

# List all jpg files in the directory
jpg_files = [f for f in os.listdir(directory) if f.endswith(".jpg")]

# Rename files sequentially
for index, file in enumerate(jpg_files, start=1):
    old_path = os.path.join(directory, file)
    new_name = f"test{count}.jpg"  # Change the naming format if needed
    new_path = os.path.join(directory, new_name)
    
    os.rename(old_path, new_path)
    print(f"Renamed: {file} -> {new_name}")
    count = count + 1

print("All files renamed successfully!")

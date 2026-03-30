from pathlib import Path

# Find pill directories
pill_dirs = list(Path('.').glob('**/pill'))
print("📁 Pill directories found:")
for d in pill_dirs:
    jpg_count = len(list(d.glob('*.jpg')))
    print(f"   {d}: {jpg_count} jpg files")

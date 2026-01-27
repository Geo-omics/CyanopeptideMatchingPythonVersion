#group_run_files
def group_files_by_timestamp(base_dir=".", minutes=5, custom_name=None):
    """
    Group files/folders based on filesystem creation time.
    
    Output folder format:
        <custom_name>_run_<timestamp>
    or if no name:
        run_<timestamp>
    """
    import os
    import shutil
    from datetime import datetime, timedelta

    FILE_EXTS = (".csv", ".xlsx", ".png")
    EXCLUDE_DIRS = {"MS1_points"}  # <-- add more if you want


    # Collect entries (files + dirs)
    entries = []
    for name in os.listdir(base_dir):
        if name.startswith("run_"):
            continue
        path = os.path.join(base_dir, name)
        if os.path.isfile(path) and name.endswith(FILE_EXTS):
            entries.append(name)
        elif os.path.isdir(path):
            entries.append(name)

    if not entries:
        print("No candidate files or folders found.")
        return

    # Map entries → creation times
    ts_map = {
        name: datetime.fromtimestamp(os.path.getctime(os.path.join(base_dir, name)))
        for name in entries
    }

    latest_time = max(ts_map.values())
    print(f"Most recent creation time: {latest_time}")

    window = timedelta(minutes=minutes)
    selected = [
        name for name, t in ts_map.items()
        if abs(t - latest_time) <= window
    ]

    if not selected:
        print("No entries found inside the time window.")
        return

    # --- Build folder name with custom name first ---
    ts_str = latest_time.strftime("%Y-%m-%d_%H-%M-%S")
    if custom_name:
        folder_name = f"{custom_name}_run_{ts_str}"
    else:
        folder_name = f"run_{ts_str}"

    dest_dir = os.path.join(base_dir, folder_name)
    os.makedirs(dest_dir, exist_ok=True)

    # Move items
    for name in selected:
        shutil.move(os.path.join(base_dir, name), os.path.join(dest_dir, name))
        print(f"Moved: {name}")

    print(f"\nGrouped {len(selected)} items into folder: {folder_name}")
    return dest_dir

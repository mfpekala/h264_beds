import os


def delete_duplicate_files(source_folder, target_folder):
    # Get list of filenames in source folder
    source_files = os.listdir(source_folder)

    # Get list of filenames in target folder
    target_files = os.listdir(target_folder)

    # Iterate through files in target folder
    for file_name in target_files:
        # Check if file exists in source folder
        if file_name in source_files:
            file_path = os.path.join(target_folder, file_name)
            # Delete file in target folder
            os.remove(file_path)
            print(f"Deleted {file_path}")


if __name__ == "__main__":
    # Source and target folder paths
    source_folder = "i"
    target_folder = "p"

    # Call function to delete duplicate files
    delete_duplicate_files(source_folder, target_folder)

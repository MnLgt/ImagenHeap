import os
import stat

def check_permissions(path):
    """
    Check read permissions for the given path and its contents.
    
    Args:
    path (str): The path to check permissions for.
    
    Returns:
    list: A list of paths with permission issues.
    """
    issues = []
    
    if not os.path.exists(path):
        return [f"Path does not exist: {path}"]
    
    if not os.access(path, os.R_OK):
        issues.append(f"No read permission for: {path}")
    
    if os.path.isdir(path):
        try:
            for root, dirs, files in os.walk(path):
                for item in dirs + files:
                    item_path = os.path.join(root, item)
                    if not os.access(item_path, os.R_OK):
                        issues.append(f"No read permission for: {item_path}")
        except PermissionError as e:
            issues.append(f"Permission error while accessing directory: {e}")
    
    return issues

# Example usage
if __name__ == "__main__":
    dataset_path = "/home/ubuntu/SPAICE/SEGMENT/datasets"
    permission_issues = check_permissions(dataset_path)
    
    if permission_issues:
        print("Permission issues found:")
        for issue in permission_issues:
            print(f"- {issue}")
    else:
        print("No permission issues found.")
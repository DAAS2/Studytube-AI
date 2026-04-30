"""
setup_users.py — Manage StudyTube AI users
------------------------------------------
Usage:
    python setup_users.py add    <username> <display_name> <email> <password>
    python setup_users.py remove <username>
    python setup_users.py list

Examples:
    python setup_users.py add alice "Alice Smith" alice@example.com mypassword123
    python setup_users.py remove alice
    python setup_users.py list
"""

import sys
import yaml
import bcrypt
from yaml.loader import SafeLoader

CONFIG_FILE = "users.yaml"

# ── Load or create config ─────────────────────────────────────────────────────

def load_config() -> dict:
    try:
        with open(CONFIG_FILE) as f:
            return yaml.load(f, Loader=SafeLoader)
    except FileNotFoundError:
        return {
            "credentials": {"usernames": {}},
            "cookie": {
                "name": "studytube_auth",
                "key": "studytube_secret_key_change_me",
                "expiry_days": 30,
            },
        }

def save_config(config: dict):
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"✅ Saved {CONFIG_FILE}")

# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_add(username: str, display_name: str, email: str, password: str):
    config = load_config()
    users  = config["credentials"]["usernames"]

    if username in users:
        print(f"⚠️  User '{username}' already exists. Remove first if you want to reset.")
        sys.exit(1)

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[username] = {
        "name":     display_name,
        "email":    email,
        "password": hashed,
        "failed_login_attempts": 0,
        "logged_in": False,
    }
    save_config(config)
    print(f"✅ Added user: {username} ({display_name} <{email}>)")

def cmd_remove(username: str):
    config = load_config()
    users  = config["credentials"]["usernames"]

    if username not in users:
        print(f"❌ User '{username}' not found.")
        sys.exit(1)

    del users[username]
    save_config(config)
    print(f"🗑  Removed user: {username}")
    print("   Note: their video_library/<username>/ folder was NOT deleted.")

def cmd_list():
    config = load_config()
    users  = config["credentials"]["usernames"]

    if not users:
        print("No users configured yet.")
        return

    print(f"{'Username':<20} {'Display Name':<25} {'Email'}")
    print("-" * 65)
    for uname, info in users.items():
        print(f"{uname:<20} {info.get('name',''):<25} {info.get('email','')}")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(0)

    cmd = args[0].lower()

    if cmd == "add":
        if len(args) != 5:
            print("Usage: python setup_users.py add <username> <display_name> <email> <password>")
            sys.exit(1)
        cmd_add(args[1], args[2], args[3], args[4])

    elif cmd == "remove":
        if len(args) != 2:
            print("Usage: python setup_users.py remove <username>")
            sys.exit(1)
        cmd_remove(args[1])

    elif cmd == "list":
        cmd_list()

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main()
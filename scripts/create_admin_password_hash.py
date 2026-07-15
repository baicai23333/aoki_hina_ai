"""Interactively create an Argon2 hash for AOKI_ADMIN_PASSWORD_HASH."""

from getpass import getpass

from argon2 import PasswordHasher


def main() -> int:
    password = getpass("管理员密码：")
    confirmation = getpass("再次输入管理员密码：")
    if len(password) < 12:
        print("管理员密码至少需要 12 个字符。")
        return 1
    if password != confirmation:
        print("两次输入的密码不一致。")
        return 1

    print("\n将下面这一整行填入 .env：")
    print(f"AOKI_ADMIN_PASSWORD_HASH={PasswordHasher().hash(password)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

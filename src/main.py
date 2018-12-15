import sqlite3
import os


def main():
    db_path = "../data/database.sqlite"
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()



if __name__ == "__main__":
    main()

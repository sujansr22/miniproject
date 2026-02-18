import MySQLdb
from config import Config

def setup_consultant_database():
    try:
        db = MySQLdb.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            passwd=Config.MYSQL_PASSWORD,
            db=Config.MYSQL_DB
        )
        cur = db.cursor()
        
        # Read the SQL file
        with open('consultant_schema.sql', 'r') as f:
            sql_script = f.read()
        
        # Split script into individual commands (assuming simple commands separated by ;)
        commands = sql_script.split(';')
        
        for command in commands:
            if command.strip():
                print(f"Executing: {command[:50]}...")
                cur.execute(command)
        
        db.commit()
        cur.close()
        db.close()
        print("Consultant database setup completed successfully.")
        
    except Exception as e:
        print(f"Error setting up database: {e}")

if __name__ == "__main__":
    setup_consultant_database()

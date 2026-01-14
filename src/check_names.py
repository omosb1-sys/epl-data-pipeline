import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0623os1978",
    database="epl_x_db",
    auth_plugin='mysql_native_password'
)
cursor = conn.cursor()
cursor.execute("SELECT team_name FROM clubs")
teams = cursor.fetchall()
print("📋 DB에 저장된 팀 목록:")
for t in teams:
    print(f"[{t[0]}]")
conn.close()

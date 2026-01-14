import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="0623os1978",
    database="epl_x_db",
    auth_plugin='mysql_native_password'
)
cursor = conn.cursor()
cursor.execute("SELECT id, team_name FROM clubs")
teams = cursor.fetchall()
print("📋 DB 팀 이름 원본 (ID: '이름'):")
for t_id, t_name in teams:
    # 텍스트 그대로 출력 (따옴표로 감싸서 공백 확인)
    print(f"{t_id}: '{t_name}'")
conn.close()

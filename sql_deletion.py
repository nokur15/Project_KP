import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="serenaochacohina",
  database="datacctv"
)

mycursor = mydb.cursor()

sql = "DROP TABLE IF EXISTS data"

mycursor.execute(sql)

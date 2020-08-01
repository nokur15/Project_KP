import mysql.connector

def filefotobeda(x,y):
    same = 0
    for a in y:
        if x == a:
            same += 1
    if same > 0:
        return False
    else:
        return True

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="serenaochacohina",
  database="datacctv"
)

mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM data")

myresult = mycursor.fetchall()
print(myresult)
print(myresult[0][0])

for x in myresult:
  print(x)

id3 = (1,)
id2 = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17)
#id2 = id2 + (mycursor.lastrowid,)
#id3 = id3 + (mycursor.rowcount,)

sql = "SELECT file_foto FROM data WHERE id IN (%s)" % (', '.join(str(id) for id in id2))

mycursor.execute(sql)
comp = mycursor.fetchall()

print(comp)

for x in comp:
    print(x)
print(id2)
print(id3)

inputted = False

filecomp = ("data/user.6.jpg",)
if not(inputted):
    print('ini benar')
else:
    print('ini salah')


if filefotobeda(filecomp,comp):
    print('this is right')
else:
    print('this is wrong')

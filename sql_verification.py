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

def query_mysql(query):
	cnx = mysql.connector.connect(user='root', password='serenaochacohina',
								  host='localhost',
								  database='datacctv',charset="utf8", use_unicode = True)
	cursor = cnx.cursor()
	cursor.execute(query)
	#get header and rows
	header = [i[0] for i in cursor.description]
	rows = [list(i) for i in cursor.fetchall()]
	#append header to rows
	rows.insert(0,header)
	cursor.close()
	cnx.close()
	return rows

#take list of lists as argument
def nlist_to_html(list2d):
	#bold header
	htable=u'<table border="1" bordercolor=000000 cellspacing="0" cellpadding="1" style="table-layout:fixed;vertical-align:bottom;font-size:13px;font-family:verdana,sans,sans-serif;border-collapse:collapse;border:1px solid rgb(130,130,130)" >'
	list2d[0] = [u'<b>' + i + u'</b>' for i in list2d[0]]
	for row in list2d:
		newrow = u'<tr>'
		newrow += u'<td align="left" style="padding:1px 4px">'+str(row[0])+u'</td>'
		row.remove(row[0])
		newrow = newrow + ''.join([u'<td align="right" style="padding:1px 4px">' + str(x) + u'</td>' for x in row])
		newrow += '</tr>'
		htable+= newrow
	htable += '</table>'
	return htable

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

query = "SELECT * FROM data"
hasil = nlist_to_html(query_mysql(query))
print(hasil)

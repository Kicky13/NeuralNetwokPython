from peewee import *

mysql_db = MySQLDatabase('vsm', user='root', password='blegoh',
                         host='127.0.0.1', port=3306)

class Datasets(Model):
    asal_smp = CharField()
    lebih_tua = IntegerField()
    jurusan = CharField()
    pekerjaan_ortu = CharField()
    kelas = CharField()


    class Meta:
        database = mysql_db # This model uses the "people.db" database.


grandma = Datasets.select()

for p in grandma:
    print p.kelas
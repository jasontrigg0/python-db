#!/usr/bin/env python
import pydb.utils

class MySQLdb_Engine():
    __metaclass__ = pydb.utils.Singleton
    def __init__(self):
        self.connection = MySQLdb.connect(host="ec2-52-90-165-109.compute-1.amazonaws.com",user="start",
                  passwd="2yur8tXGWQkZRaRY",db="start")

        
def run(sql):
    db = MySQLdb_Engine().connection
    cursor = db.cursor()
    cursor.execute(sql)
    return cursor.fetchall()
    
if __name__ == "__main__":
    print run("SELECT * FROM product_variations LIMIT 10")

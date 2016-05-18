#!/usr/bin/env python
import MySQLdb
import sys
import argparse
import os

import pcsv.any2csv
import pcsv.utils

import pydb.utils

class MySQLdb_Engine():
    __metaclass__ = pydb.utils.Singleton
    def __init__(self):
        self.connection = self.connect()
    @classmethod
    def read_config(self):
        mysql_config_file = get_home_directory() + "/.my.cnf"
        if not os.path.exists(mysql_config_file):
            sys.stderr.write("ERROR: can't find mysql config file. Should be located at {mysql_config_file}".format(**vars()) + "\n")
            sys.exit(-1)
        user = None; passwd = None
        with open(mysql_config_file) as f_in:
            for l in f_in:
                if l.startswith("user"):
                    user = l.strip().split("=")[1]
                if l.startswith("password"):
                    passwd = l.strip().split("=")[1]
        if user is None or passwd is None:
            sys.stderr.write("ERROR: couldn't find user or password in mysql config file. Make sure lines for user=MYUSERNAME and password=MYPASSWORD both exist in the file" + "\n")
            sys.exit(-1)
        return user, passwd
    @classmethod
    def connect(self):
        if not "MYSQL_DB" in os.environ:
            sys.stderr.write('ERROR: can\'t find MYSQL_DB variable! Set with \'export MYSQL_DB="my_db_name"\' or equivalent' + "\n")
            sys.exit(-1)
        if not "MYSQL_HOST" in os.environ:
            sys.stderr.write('ERROR: can\'t find MYSQL_HOST variable! Set with \'export MYSQL_HOST="my_mysql_host"\' or equivalent' + "\n")
            sys.exit(-1)
        db = os.environ["MYSQL_DB"]
        host = os.environ["MYSQL_HOST"]
        user, passwd = self.read_config()
        return MySQLdb.connect(host=host,user=user,passwd=passwd,db=db)
        
            
def get_home_directory():
    from os.path import expanduser
    home = expanduser("~")
    return home

def get_tables():
    df = pcsv.any2csv.csv2df(run("SHOW tables"))
    return [r[0] for r in df.values]
    
def lookup_table_abbreviation(abbrev):
    table_list = get_tables()
    if abbrev in table_list:
        return abbrev
    for table in table_list:
        table_abbreviation = "".join([t[0] for t in table.split("_")])
        if abbrev == table_abbreviation:
            return table
    return None

def run(sql, df=False):
    return run_list([sql], df=False)

def run_list(sql_list, df=False):
    """
    run sql and return either df of results or a string
    """
    db = MySQLdb_Engine().connection
    cursor = db.cursor()
    pid = db.thread_id()
    try:
        for s in sql_list:
            cursor.execute(s)
    except (KeyboardInterrupt, SystemExit):
        new_cursor = MySQLdb_Engine.connect().cursor() #old cursor/connection is unusable
        new_cursor.execute('KILL QUERY ' + str(pid))
        sys.exit()
    db.commit()
    if cursor.description:
        field_names = [i[0] for i in cursor.description]
        rows = [field_names] + list(cursor.fetchall())
        rows = [[process_field(i) for i in r] for r in rows]
        csv = pcsv.any2csv.rows2csv(rows)
        if df:
            return pcsv.any2csv.csv2df(csv)
        else:
            return csv

def process_field(f):
    if f is None:
        return "NULL"
    else:
        return str(f)
    
def readCL():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--index",action="store_true",help="show indexes on a table")
    parser.add_argument("-d","--describe",action="store_true",help="describe table")
    parser.add_argument("--cat",action="store_true")
    parser.add_argument("--head",action="store_true")
    parser.add_argument("--tail",action="store_true")
    parser.add_argument("-r","--raw",action="store_true",help="print raw csv instead of pretty printing")
    parser.add_argument("-a","--show_all",action="store_true",help="print entire fields regardless of width")
    parser.add_argument("--top",action="store_true",help="show currently running processes")
    parser.add_argument("-p","--profile",action="store_true",help="profile the given query")
    parser.add_argument("-k","--kill")
    parser.add_argument("-t","--table", action="store_true", help="db -t table_name col1 col2... --> frequencies for col1,col2 in table_name")
    parser.add_argument("-w","--where",action="store_true",help="db -w table_name col val --> 'SELECT * FROM table_name WHERE col = val'")
    parser.add_argument("positional",nargs="*")
    args = parser.parse_args()
    if args.top:
        args.show_all = True
    return args.index, args.describe, args.cat, args.head, args.tail, args.show_all, args.top, args.kill, args.profile, args.where, args.table, args.raw, args.positional

def main():
    index, describe, cat, head, tail, show_all, top, kill, profile, where, freq, raw, pos = readCL()
    if any([index, describe, cat, head, tail, where, freq]):
        lookup = lookup_table_abbreviation(pos[0])
        if lookup:
            table = lookup
        else:
            table = pos[0]
    if kill:
        out = run("KILL QUERY {kill}".format(**vars()))
    elif top:
        out = run("SHOW FULL PROCESSLIST")
    elif index:
        out = run("SHOW INDEX FROM {table}".format(**vars()))
    elif describe:
        out = run("DESCRIBE {table}".format(**vars()))
    elif cat:
        out = run("SELECT * FROM {table}".format(**vars()))
    elif head:
        out = run("SELECT * FROM {table} LIMIT 10".format(**vars()))
    elif tail:
        cnt = run("SELECT count(*) FROM {table}".format(**vars()), df=True)
        cnt = cnt.iloc[0,0]
        offset = int(cnt) - 10
        out = run("SELECT * FROM {table} LIMIT {offset},10".format(**vars()))
    elif where:
        out = run("SELECT * FROM {table} WHERE {pos[1]} = '{pos[2]}'".format(**vars()))
    elif freq:
        csv = ",".join(pos[1:])
        out = run("SELECT {csv},count(*) FROM {table} GROUP BY {csv}".format(**vars()))
    else:
        if profile:
            out = run_list(['SET profiling = 1;'] + pos + ["SHOW PROFILE"])
        else:
            out = run_list(pos)

    if show_all:
        max_field_size = None
    else:
        max_field_size = 50

    if raw:
        sys.stdout.write(out + "\n")
    else:
        out = pcsv.any2csv.csv2pretty(out,max_field_size)
        pcsv.utils.lines2less(out.split("\n"))

if __name__ == "__main__":
    main()

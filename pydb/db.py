#!/usr/bin/env python2
#profiling results:
#total time = 0.35 - 0.4 (slower on the first run)
#0.08 - 0.09 for SHOW TABLES to lookup abbreviations
#0.08 - 0.09 for main query
#(half of the query time for the main query, second half for the commit)
#0.11 for imports (note: jtutils imports were slow before, may want to watch this)
#~0.1 for the rest
#if database needs a reconnect that's 0.3-0.4ish? but this is usually avoided by the server

import MySQLdb
import sys
import argparse
import os
import re

import jtutils
import pcsv.any2csv

import pydb.utils
import pyservice

import math

class MySQLdb_Engine(object): #, metaclass=pydb.utils.Singleton):
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
    x = run("SHOW tables")
    df = pcsv.any2csv.csv2df(x)
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

def test_delete_query(s, params):
    delete_start = "^[ ]*(DELETE|delete) "
    if re.findall(delete_start,s):
        #DELETE syntax: https://dev.mysql.com/doc/refman/5.7/en/delete.html
        delete_clause = "^[ ]*(DELETE|delete) .*?(FROM|from)" #.*? to pull the first instance of FROM
        if not re.findall(delete_clause, s.upper()):
            raise Exception("Unusual delete syntax".format(s))
        query = re.sub(delete_clause,"",s)
        query = re.sub(".*\\b(USING|using)\\b","",query)
        #wonky syntax to make limit statements work
        #or else SELECT COUNT(*) FROM blah LIMIT 5 isn't limited by 5
        #https://stackoverflow.com/q/17020842
        select_query = "SELECT COUNT(*) FROM (SELECT 1 FROM "+query+") as a"
        try:
            out = run(select_query, df=True, params=params)
        except:
            # sys.stderr.write("WARNING: unable to run row count query.")
            # return True
            return jtutils.y_n_input("WARNING: unable to run row count query. Do you want to continue? [y/n]\n")
        cnt = out.values[0][0]
        if cnt >= 10:
            return jtutils.y_n_input("WARNING: this command will delete {} rows. Do you want to continue? [y/n]\n".format(cnt))
        else:
            return True
    return True

def test_update_query(s, params):
    update_start = "^[ ]*(UPDATE|update) "
    if re.findall(update_start,s):
        #UPDATE syntax: https://dev.mysql.com/doc/refman/5.7/en/update.html
        update_clause = "^[ ]*(UPDATE|update) .*?(SET|set)" #.*? to pull the first instance of SET
        if not re.findall(update_clause, s.upper()):
            raise Exception("Unusual update syntax: {}".format(s))
        query = re.sub("\\b(LOW_PRIORITY|low_priority)\\b","",s)
        query = re.sub("\\b(IGNORE|ignore)\\b","",query)
        query = re.sub("\\b(SET|set)\\b.*?(WHERE|where|ORDER BY|order by|LIMIT|limit|$)","\\2",query)
        query = re.sub("\\bUPDATE\\b","",query)
        #wonky syntax to make limit statements work
        #or else SELECT COUNT(*) FROM blah LIMIT 5 isn't limited by 5
        #https://stackoverflow.com/q/17020842
        select_query = "SELECT COUNT(*) FROM (SELECT 1 FROM "+query+") as a"
        try:
            out = run(select_query, df=True, params=params)
        except:
            #sys.stderr.write("WARNING: unable to run row count query.")
            #return True
            return jtutils.y_n_input("WARNING: unable to run row count query. Do you want to continue? [y/n]\n")
        cnt = out.values[0][0]
        if cnt >= 10:
            return jtutils.y_n_input("WARNING: this command will update {} rows. Do you want to continue? [y/n]\n".format(cnt))
        else:
            return True
    return True

def run(sql, df=False, params=None):
    return run_list([sql], df, [params])

def run_list(sql_list, df=False, params_list=None):
    """
    run sql and return either df of results or a string
    """
    db = MySQLdb_Engine().connection
    cursor = db.cursor()
    pid = db.thread_id()
    try:
        for i,s in enumerate(sql_list):
            params = params_list[i] if params_list else None
            if not test_delete_query(s, params): return
            if not test_update_query(s, params): return
            cursor.execute(s,params)
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

def foreign_key_graph(include_nullable = False):
    #@returns [(origin_table, origin_column),(destination_table, destination_column)]
    #@example:
    #[(('outbound_emails', 'issuing_entity_id'), ('issuing_entities', 'id')), (('downloads', 'document_file_id'), ('document_files', 'id')), (('account_activity_documents', 'activity_id'), ('account_activities', 'id')), (('BACKUP_linked_firms', 'investing_entity_id'), ('investing_entities', 'id')), (('investments', 'type_id'), ('investment_types', 'id'))]
    fields = get_fields()
    isNullable = {(x[0],x[1]):1*(x[2] == "YES") for x in fields[["TABLE_NAME","COLUMN_NAME","IS_NULLABLE"]].values}
    df = run('SELECT k.TABLE_NAME, k.COLUMN_NAME, i.CONSTRAINT_TYPE, i.CONSTRAINT_NAME, k.REFERENCED_TABLE_NAME, k.REFERENCED_COLUMN_NAME FROM information_schema.TABLE_CONSTRAINTS i LEFT JOIN information_schema.KEY_COLUMN_USAGE k ON i.CONSTRAINT_NAME = k.CONSTRAINT_NAME WHERE i.CONSTRAINT_TYPE = "FOREIGN KEY" AND i.TABLE_SCHEMA = DATABASE()', df=True)
    if include_nullable:
        graph = df[["TABLE_NAME","COLUMN_NAME","REFERENCED_TABLE_NAME","REFERENCED_COLUMN_NAME"]].values
    else:
        graph = df[["TABLE_NAME","COLUMN_NAME","REFERENCED_TABLE_NAME","REFERENCED_COLUMN_NAME"]][df.apply(lambda x: not isNullable[(x[0],x[1])], axis = 1)].values
    graph = [((x[0],x[1]),(x[2],x[3])) for x in graph]
    graph = list(set(graph)) #dedup
    return graph

def get_fields():
    return run('select TABLE_NAME, COLUMN_NAME, ORDINAL_POSITION, COLUMN_DEFAULT, IS_NULLABLE, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, COLUMN_KEY from information_schema.columns where table_schema = DATABASE() order by table_name, ordinal_position', df=True)

def merge_fn(table, where_clause, row_dict):
    #merge all rows with table satisfying where_clause
    #to the row with values row_dict
    #(requires id column in table)
    graph = foreign_key_graph()
    deps = [g for g in graph if g[1][0] == table]

    if not "id" in row_dict:
        raise Exception("Merging requires an id column")

    query = 'UPDATE {table}'.format(**vars())
    joins = []
    sets = []
    where = ""

    #TODO: print the rows and then confirm before proceeding
    print(run('SELECT * FROM '+table+' WHERE id = ' + str(row_dict["id"])))
    print(run('SELECT * FROM {table} WHERE {where_clause}'.format(**vars())))

    for d in deps:
        start, end = d
        start_table, start_col = start
        end_table, end_col = end
        joins.append(' LEFT OUTER JOIN {start_table} ON {end_table}.{end_col} = {start_table}.{start_col}'.format(**vars()))
        new_val = row_dict[end_col]
        sets.append(' {start_table}.{start_col} = {new_val}'.format(**vars()))


    update_query = query + ''.join(joins) + " SET " + ','.join(sets) + " WHERE " + where_clause
    delete_query = "DELETE FROM {table} WHERE {where_clause} AND NOT {table}.id = ".format(**vars())+str(row_dict["id"])

    print(update_query)
    print(delete_query)
    ok = jtutils.y_n_input('Do you want to merge the above rows? [y/n]\n')
    if not ok: return

    run(update_query)
    run(delete_query)


def downstream_query(tables, select_clause=None, show_all=False):
    fields = get_fields()

    graph = foreign_key_graph()
    foreign_key_columns = set(x[0] for x in graph) #foreign_key_columns = {("table1","col1")}


    table_to_columns = {}
    for table, column, is_nullable, data_type, column_key in fields[["TABLE_NAME","COLUMN_NAME","IS_NULLABLE","DATA_TYPE","COLUMN_KEY"]].values:
        table_to_columns.setdefault(table,[])
        if show_all: #view all columns
            table_to_columns[table].append(column)
            continue
        if is_nullable == "YES": continue #skip nullable columns
        if (table,column) in foreign_key_columns: continue #skip foreign keys
        #print id columns that aren't foreign keys
        if column_key in ["PRI","MUL"]:
            table_to_columns[table].append(column)
        if data_type not in ["enum","varchar"]: continue
        table_to_columns[table].append(column)

    todo_up = set()
    # todo_up.add("linked_issuing_entities")
    #out = run("SELECT * FROM investing_entities");

    start_table = tables[0]
    tables_to_join = tables[1:]
    #join on the tables_to_join even if it requires one step upstream
    #eg: if the table graph is contacts <--- firm_contacts ---> firms
    #if tables = [contacts, firms]
    #then start_table = contacts, tables_to_join = [firms]
    #then we'll join one step upstream to firm_contacts so we can include firms
    #intermediates = [firm_contacts]
    intermediates = {}
    for start,end in graph:
        if start[0] not in tables_to_join and end[0] in tables_to_join:
            intermediates[start[0]] = end[0]

    def field_name(table, column, show_all):
        if show_all:
            return table + "." + column + " as " + table + "$" + column
        else:
            return table + "." + column + " as " + "".join([x[0] for x in table.split("_")])+"$"+column
    select_cols = [field_name(start_table,col,show_all) for col in table_to_columns[start_table]]
    body = "FROM {start_table}".format(**vars())
    todo = set()
    done = set()
    todo.add(start_table)
    while len(todo) > 0:
        table = todo.pop()
        for start,end in graph:
            start_table = start[0]
            start_col = start[1]
            end_table = end[0]
            end_col = end[1]
            if start_table == table:
                if end_table in todo.union(done).union(set([table])):
                    #print(f"repeat: {start_table} -> {end_table}")
                    continue #already joined on this
                select_cols += [field_name(end_table,col,show_all) for col in table_to_columns[end_table]]
                body += ' LEFT OUTER JOIN {end_table} ON {end_table}.{end_col} = {start_table}.{start_col}'.format(**vars())
                #print('ADDING: {end_table}'.format(**vars()))
                todo.add(end_table)
            elif start_table in intermediates and intermediates[start_table] not in todo.union(done).union(set([table])) and end_table == table:
                if start_table in todo.union(done).union(set([table])):
                    continue #already joined on this
                select_cols += [field_name(start_table,col,show_all) for col in table_to_columns[start_table]]
                body += ' LEFT OUTER JOIN {start_table} ON {end_table}.{end_col} = {start_table}.{start_col}'.format(**vars())
                #print('ADDING INTERMEDIATE: {start_table} for help with {intermediates[start_table]}'.format(**vars()))
                #print(intermediates[start_table])
                todo.add(start_table)
        done.add(table)
    for t in tables_to_join:
        if not t in done:
            raise Exception("Couldn't join from {start_table} to {t}. Can you add an intermediate table to help?".format(**vars()))
    if not select_clause:
        select_clause = "SELECT " + ",".join(select_cols)
    query = select_clause + " " + body
    #print(done)
    return query


def gen_tree(reverse):
    #print a tree of tables like the tree command
    """
    .
    ├── bak.babelrc
    ├── build
    │   ├── asset-manifest.json
    │   ├── marketing
    │   │   ├── favicon.ico

    """

    all_nodes = set(get_tables())
    fk_graph_list = foreign_key_graph()
    if reverse:
        table_graph_list = list(set([(x1[0],x2[0]) for x1,x2 in fk_graph_list if x1[0] != x2[0]])) #remove self-edges and dups
    else:
        table_graph_list = list(set([(x2[0],x1[0]) for x1,x2 in fk_graph_list if x1[0] != x2[0]])) #remove self-edges and dups
    table_graph_dict = {}
    reverse_table_graph_dict = {}
    for t1,t2 in table_graph_list:
        table_graph_dict.setdefault(t1,[]).append(t2)
        reverse_table_graph_dict.setdefault(t2,[]).append(t1)
    for t in all_nodes:
        table_graph_dict.setdefault(t,[])
        reverse_table_graph_dict.setdefault(t,[])

    #taking the root of the tree as the node with no outgoing edges
    #(this is opposite of the regular convention?)
    root_nodes = []
    for t in all_nodes:
        if len(reverse_table_graph_dict[t]) == 0:
            root_nodes.append(t)

    #add all nodes downstream from root_nodes to done
    todo = root_nodes[:]
    done = set()
    while len(todo) > 0:
        t = todo.pop()
        for child in table_graph_dict[t]:
            if child in done: continue
            todo.append(child)
        done.add(t)

    #grab the remaining components (may not be trees) one at a time
    for t in all_nodes:
        if t in done: continue
        todo = [t]
        component = []
        #compute this component
        while len(todo) > 0:
            t = todo.pop()
            if t in component: continue
            for child in table_graph_dict[t]:
                todo.append(child)
            component.append(t)
        root = sorted(component, key=lambda x: len(table_graph_dict[x]), reverse=True)[0]
        root_nodes.append(root)
        done = done.union(set(component))


    def prefix(depth, active_depths):
        out = ""
        if depth == 0: return out
        for d in active_depths[:-1]:
            if d:
                out +=  "│   "
            else:
                out +=  "    "
        if active_depths[-1]:
            out += "├── "
        else:
            out += "└── "
        return out

    def tree(node, depth, active_depths): #active_depths: levels of the tree that have more children remaining
        out = prefix(depth, active_depths) + node + '\n'
        if node in done:
            return out
        else:
            done.add(node)
            children = list(table_graph_dict[node])
            children.sort(key = lambda x: len(table_graph_dict[x]),reverse=True)
            for i,child in enumerate(children):
                if i == len(children) - 1: #last child
                    out += tree(child, depth+1, active_depths + [0])
                else:
                    out += tree(child, depth+1, active_depths + [1])
        return out

    out = ""
    #print trees, starting from the table with the most incoming edges
    for k in sorted(root_nodes, key=lambda x: len(table_graph_dict[x]), reverse=True):
        done = set()
        out += (tree(k,0,[]))
        out += "\n\n"

    return out


def readCL(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--index",action="store_true",help="show indexes on a table")
    parser.add_argument("-d","--describe",action="store_true",help="describe table")
    parser.add_argument("--cat",action="store_true")
    parser.add_argument("--head",action="store_true")
    parser.add_argument("--tail",action="store_true")
    parser.add_argument("--top",action="store_true",help="show currently running processes")
    parser.add_argument("-p","--profile",action="store_true",help="profile the given query")
    parser.add_argument("-k","--kill")
    parser.add_argument("-t","--table", action="store_true", help="db -t table_name col1 col2... --> frequencies for col1,col2 in table_name")
    parser.add_argument("-w","--where",action="store_true",help="db -w table_name col val --> 'SELECT * FROM table_name WHERE col = val'")
    parser.add_argument("--key",action="store_true", help="db --key table_name primary_key_value --> 'SELECT * FROM table_name WHERE primary_key = primary_key_value'")
    parser.add_argument("--tree",action="store_true",help="view tree(s) of foreign key dependencies between tables")
    parser.add_argument("--tree_rev",action="store_true",help="view reversed tree, for help when deleting tables")
    parser.add_argument("--cascade_select",nargs="*",help="starting from one table, join all tables referred to by foreign keys")
    parser.add_argument("--cascade_select_all",nargs="*",help="like --cascade_select_all, but shows all columns instead of select columns")
    parser.add_argument("--cascade_select_query",nargs="*",help="print query to start from one table and join all tables referred to by foreign keys")
    parser.add_argument("--merge_dups",nargs=2,help="db --merge_dups users first_name,last_name")
    parser.add_argument("--merge",nargs=3,help="merge all dependencies on one row from a table to point to another row from that table. example to switch everything pointing to user 210 -> user 217: db --merge users 'users.id = 210' 'users.id = 217'")
    parser.add_argument("-j","--join",help="automatically make the most logical join to the argument table based on foreign keys")
    parser.add_argument("-s","--select",help="override the default select statement in, eg cascade_select")
    parser.add_argument("positional",nargs="*")
    if args:
        args, _ = parser.parse_known_args(args[1:]) #args[0] is the script name
    else:
        args, _ = parser.parse_known_args()
    if args.top:
        args.show_all = True
    return args.index, args.describe, args.cat, args.head, args.tail, args.top, args.kill, args.profile, args.where, args.key, args.table, args.positional, args.tree, args.tree_rev, args.cascade_select, args.cascade_select_all, args.cascade_select_query, args.merge_dups, args.merge, args.join, args.select

def readCL_output():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r","--raw",action="store_true",help="print raw csv instead of pretty printing")
    parser.add_argument("-a","--show_all",action="store_true",help="print entire fields regardless of width")
    args, _ = parser.parse_known_args()
    return args.raw, args.show_all


def execute_query(args):
    index, describe, cat, head, tail, top, kill, profile, where, key, freq, pos, tree, tree_rev, cascade_select, cascade_select_all, cascade_select_query, merge_dups, merge, join, select = readCL(args)
    if any([index, describe, cat, head, tail, where, key, freq]):
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
        #TODO: support tail values != 10
        cnt = run("SELECT count(*) FROM {table}".format(**vars()), df=True)
        cnt = cnt.iloc[0,0]
        offset = int(cnt) - 10
        out = run("SELECT * FROM {table} LIMIT {offset},10".format(**vars()))
    elif where:
        out = run("SELECT * FROM {table} WHERE {pos[1]} = %s".format(**vars()), params = (pos[2],))
    elif key:
        key = run("SHOW INDEX FROM {table} WHERE Key_name = 'PRIMARY'".format(**vars()), df=True)["Column_name"].values[0]
        # function dbk() { col=$(db -r "SHOW INDEX FROM $1 WHERE Key_name = 'PRIMARY'" | pcsv -c 'Column_name' | pawk -g 'i==1'); db "SELECT * FROM $1 WHERE $col = $2"; }
        out = run("SELECT * FROM {table} WHERE {key} = %s".format(**vars()), params = (pos[1],))
    elif freq:
        csv = ",".join(pos[1:])
        out = run("SELECT {csv},count(*) FROM {table} GROUP BY {csv}".format(**vars()))
    elif tree:
        out = gen_tree(False)
    elif tree_rev:
        out = gen_tree(True)
    elif cascade_select:
        query = downstream_query(cascade_select[:1], select) #TODO: fix up downstream query to use graph logic more clearly
        if len(cascade_select) > 1:
            query += " " + cascade_select[1]
        out = run(query)
    elif cascade_select_all:
        query = downstream_query(cascade_select_all[:1], select, True)
        if len(cascade_select_all) > 1:
            query += " " + cascade_select_all[1]
        out = run(query)
    elif cascade_select_query:
        out = downstream_query(cascade_select_query, select)
    elif merge_dups:
        table, cols = merge_dups
        graph = foreign_key_graph()
        duplicate_groups = run('SELECT * FROM {table} GROUP BY {cols} HAVING count(*) > 1'.format(**vars()),df=True)
        for r2 in duplicate_groups.to_dict('records'):
            where_clause = " AND ".join([table+"."+str(c)+" = "+str(r2[c]) for c in cols.split(",")])
            if any([math.isnan(r2[c]) for c in cols.split(",")]):
                print("WARNING: nan detected. Skipping...")
                print(r2)
                continue
            merge_fn(table, where_clause, r2)
    elif merge:
        table, where1, where2 = merge
        if not table in where1 or not table in where2:
            raise Exception('merge where clauses should include the name of the table: {table}'.format(**vars()))

        r2 = run('SELECT * FROM {table} WHERE {where2}'.format(**vars()),df=True)
        if len(r2) > 1:
            raise Exception('merge SELECT clause should return only a single row')
        r2 = r2.to_dict('records')[0]
        merge_fn(table, where1, r2)
    else:
        if profile:
            out = run_list(['SET profiling = 1;'] + pos + ["SHOW PROFILE"])
        else:
            out = run_list(pos)
    return out


def display_output(out):
    raw, show_all = readCL_output()
    if show_all:
        max_field_size = None
    else:
        max_field_size = 50

    if not out:
        return

    if raw:
        sys.stdout.write(out + "\n")
    else:
        out = pcsv.any2csv.csv2pretty(out,max_field_size)
        jtutils.lines2less(out.split("\n"))

def main():
    USE_SERVICE = False
    if USE_SERVICE:
        pyservice.Service('python-db', daemon_main = execute_query, client_receiver = lambda x: display_output(x)).run()
    else:
        out = execute_query(sys.argv)
        display_output(out)

if __name__ == "__main__":
    main()

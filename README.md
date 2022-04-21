# tfg_scripts

  1.extract.py
-----------------------
Error en el método para guardar el Submission en la BD. Error al ejecutar la query que tiene como objetivo verificar si el el Submission ya estaba en la BD. Traceback:

```
Traceback (most recent call last):
  File "/home/manuel/Documentos/TFG/tfg_scripts/app/extract.py", line 73, in <module>
    saveSubmission(reddit.submission(submission_praw))
  File "/home/manuel/Documentos/TFG/tfg_scripts/app/keeper/reddit_to_db.py", line 54, in saveSubmission
    result = db.session.execute(query).fetchone()  # devuelve uno
  File "/home/manuel/Documentos/TFG/tfg_scripts/venv-tfg/lib/python3.10/site-packages/sqlalchemy/orm/session.py", line 1692, in execute
    result = conn._execute_20(statement, params or {}, execution_options)
  File "/home/manuel/Documentos/TFG/tfg_scripts/venv-tfg/lib/python3.10/site-packages/sqlalchemy/engine/base.py", line 1631, in _execute_20
    return meth(self, args_10style, kwargs_10style, execution_options)
  File "/home/manuel/Documentos/TFG/tfg_scripts/venv-tfg/lib/python3.10/site-packages/sqlalchemy/sql/elements.py", line 325, in _execute_on_connection
    return connection._execute_clauseelement(
  File "/home/manuel/Documentos/TFG/tfg_scripts/venv-tfg/lib/python3.10/site-packages/sqlalchemy/engine/base.py", line 1498, in _execute_clauseelement
    ret = self._execute_context(
  File "/home/manuel/Documentos/TFG/tfg_scripts/venv-tfg/lib/python3.10/site-packages/sqlalchemy/engine/base.py", line 1862, in _execute_context
    self._handle_dbapi_exception(
  File "/home/manuel/Documentos/TFG/tfg_scripts/venv-tfg/lib/python3.10/site-packages/sqlalchemy/engine/base.py", line 2047, in _handle_dbapi_exception
    util.raise_(exc_info[1], with_traceback=exc_info[2])
  File "/home/manuel/Documentos/TFG/tfg_scripts/venv-tfg/lib/python3.10/site-packages/sqlalchemy/util/compat.py", line 207, in raise_
    raise exception
  File "/home/manuel/Documentos/TFG/tfg_scripts/venv-tfg/lib/python3.10/site-packages/sqlalchemy/engine/base.py", line 1819, in _execute_context
    self.dialect.do_execute(
  File "/home/manuel/Documentos/TFG/tfg_scripts/venv-tfg/lib/python3.10/site-packages/sqlalchemy/engine/default.py", line 732, in do_execute
    cursor.execute(statement, parameters)
  File "/home/manuel/Documentos/TFG/tfg_scripts/venv-tfg/lib/python3.10/site-packages/mysql/connector/cursor_cext.py", line 257, in execute
    prepared = self._cnx.prepare_for_mysql(params)
  File "/home/manuel/Documentos/TFG/tfg_scripts/venv-tfg/lib/python3.10/site-packages/mysql/connector/connection_cext.py", line 684, in prepare_for_mysql
    result[key] = self._cmysql.convert_to_mysql(value)[0]
_mysql_connector.MySQLInterfaceError: Python type Submission cannot be converted
```

  2.tf-idf_vec.py
-----------------------
No encuentro en qué punto del desarrollo se forma la matriz con la información que necesitamos.
Esta la necesito para guardarla en un df con el id y el flair de la publicación, y en otro script probaría a entrenar con los distintos modelos comentados tomando como partida el df anterior.

from sqlalchemy import MetaData
from sqlalchemy_schemadisplay import create_schema_graph

# Database
host     = 'localhost'
engine   = 'sqlite'
database = ''
username = ''
password = ''

# General
data_types = False
indexes    = False


# Generation
dsn = engine + ':///forjar.db';

graph = create_schema_graph(
      metadata       = MetaData(dsn),
      show_datatypes = data_types,
      show_indexes   = indexes
)
print 'Writing schema...'
graph.write_png('schema.png')